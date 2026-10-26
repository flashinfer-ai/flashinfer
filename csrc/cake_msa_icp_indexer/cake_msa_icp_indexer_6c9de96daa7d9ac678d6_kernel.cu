/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_msa_icp_indexer_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SURV_K_OFF 0
#define SMEM_SURV_K_STAGE_BYTES 128
#define SMEM_SURV_K_STRIDE 128
#define SMEM_SURV_I_OFF 128
#define SMEM_SURV_I_STAGE_BYTES 128
#define SMEM_SURV_I_STRIDE 128
#define SMEM_SORT_K_OFF 256
#define SMEM_SORT_K_STAGE_BYTES 128
#define SMEM_SORT_K_STRIDE 128
#define SMEM_SORT_I_OFF 384
#define SMEM_SORT_I_STAGE_BYTES 128
#define SMEM_SORT_I_STRIDE 128
#define SMEM_BND_K_OFF 512
#define SMEM_BND_K_STAGE_BYTES 4
#define SMEM_BND_K_STRIDE 4
#define SMEM_BND_I_OFF 516
#define SMEM_BND_I_STAGE_BYTES 4
#define SMEM_BND_I_STRIDE 4
#define SMEM_CNT_W_OFF 520
#define SMEM_CNT_W_STAGE_BYTES 4
#define SMEM_CNT_W_STRIDE 4
#define SMEM_CARRY_K_OFF 524
#define SMEM_CARRY_K_STAGE_BYTES 64
#define SMEM_CARRY_K_STRIDE 64
#define SMEM_CARRY_I_OFF 588
#define SMEM_CARRY_I_STAGE_BYTES 64
#define SMEM_CARRY_I_STRIDE 64
#define SMEM_TOTAL 768
#define THREADS 32

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_msa_icp_indexer_6c9de96daa7d9ac678d6(int* __restrict__ pool, int* __restrict__ row_meta, int* __restrict__ query_start_loc, int* __restrict__ candidates, int* __restrict__ status, int pool_stride, int request_count, int* __restrict__ forced_block, int* __restrict__ n_ordinary, int* __restrict__ out, int* __restrict__ gen_ctr, unsigned long long* __restrict__ stamps, int x_par_stride, int rank, int token_begin, long long timeout_ns, int x_phase, int32_t pg_world, int32_t pg_rank, CakePeerPointerTable<unsigned int> pg_flags, unsigned long long* __restrict__ xpool, CakePeerPointerTable<unsigned long long> xpool_peers)
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
    unsigned int* surv_k = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SURV_K_OFF);
    const int surv_k_addr = smem + SMEM_SURV_K_OFF;
    int* surv_i = reinterpret_cast<int*>(smem_raw + SMEM_SURV_I_OFF);
    const int surv_i_addr = smem + SMEM_SURV_I_OFF;
    unsigned int* sort_k = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SORT_K_OFF);
    const int sort_k_addr = smem + SMEM_SORT_K_OFF;
    int* sort_i = reinterpret_cast<int*>(smem_raw + SMEM_SORT_I_OFF);
    const int sort_i_addr = smem + SMEM_SORT_I_OFF;
    unsigned int* bnd_k = reinterpret_cast<unsigned int*>(smem_raw + SMEM_BND_K_OFF);
    const int bnd_k_addr = smem + SMEM_BND_K_OFF;
    int* bnd_i = reinterpret_cast<int*>(smem_raw + SMEM_BND_I_OFF);
    const int bnd_i_addr = smem + SMEM_BND_I_OFF;
    int* cnt_w = reinterpret_cast<int*>(smem_raw + SMEM_CNT_W_OFF);
    const int cnt_w_addr = smem + SMEM_CNT_W_OFF;
    unsigned int* carry_k = reinterpret_cast<unsigned int*>(smem_raw + SMEM_CARRY_K_OFF);
    const int carry_k_addr = smem + SMEM_CARRY_K_OFF;
    int* carry_i = reinterpret_cast<int*>(smem_raw + SMEM_CARRY_I_OFF);
    const int carry_i_addr = smem + SMEM_CARRY_I_OFF;

    // === Task calls (dependency order) ===
    int row = (int)blockIdx.x;
    int sub = (int)warp;
    int lane_0 = (int)lane;
    int tr = sub * 32 + lane_0;
    int t_end = query_start_loc[request_count];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int grow = token_begin * 4 + row;
    int g = gen_ctr[grow];
    unsigned long long gtimer_1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_1) :: "memory");
    unsigned long long t0 = gtimer_1;
    int covered = (int)(t_end > row / 4);
    long long r_base = (long long)row * (long long)pool_stride * 2;
    long long c_base = r_base;
    int S = 0;
    int rnk = 0;
    int h_real = 0;
    unsigned int hk[1];
    int hi[1];
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int e_row = row_meta[row];
    int n_e = 0;
    int n_ch = 1;
    int c = 0;
    int more = 1;
    while (more != 0) {
        int wv[4];
        {
            const int4* _ivptr_0 = reinterpret_cast<const int4*>(pool + (c_base + (long long)tr * 4) + 0);
            int4 _ivld_0;
            _ivld_0 = *_ivptr_0;
            wv[0 + 0] = _ivld_0.x;
            wv[0 + 1] = _ivld_0.y;
            wv[0 + 2] = _ivld_0.z;
            wv[0 + 3] = _ivld_0.w;
        }
        c_base = c_base + 128;
        if (c == 0) {
            n_e = ((covered != 0) ? e_row : 0);
            if (covered != 0 && e_row > pool_stride) {
                n_e = 0;
                if (tr == 0) {
                    status[0] = 2;
                }
            }
            n_ch = (n_e + 63) / 64;
            n_ch = ((n_ch > 0) ? n_ch : 1);
        }
        int fast = 0;
        fast = (int)(c == 0 && n_e <= 64);
        if (fast == 0) {
            int e_ch = n_e - c * 64;
            unsigned int kb[3];
            int ib[3];
            unsigned int kw = 0;
            kw = reinterpret_cast<unsigned int*>(&wv[0])[0];
            kb[0] = ((e_ch > 2 * tr) ? kw : 8388607);
            ib[0] = ((e_ch > 2 * tr) ? wv[1] : 2147483647);
            kw = reinterpret_cast<unsigned int*>(&wv[2])[0];
            kb[1] = ((e_ch > 1 + 2 * tr) ? kw : 8388607);
            ib[1] = ((e_ch > 1 + 2 * tr) ? wv[3] : 2147483647);
            int has_c = (int)(tr < 16 && c > 0);
            kb[2] = ((has_c != 0) ? carry_k[tr & 15] : 8388607);
            ib[2] = ((has_c != 0) ? carry_i[tr & 15] : 2147483647);
            unsigned int mk[1];
            int mi[1];
            mk[0] = kb[0];
            mi[0] = ib[0];
            int r = (int)(kb[1] > mk[0] || kb[1] == mk[0] && ib[1] < mi[0]);
            int tk = r;
            mk[0] = ((tk != 0) ? kb[1] : mk[0]);
            mi[0] = ((tk != 0) ? ib[1] : mi[0]);
            int r_0 = (int)(kb[2] > mk[0] || kb[2] == mk[0] && ib[2] < mi[0]);
            int tk_1 = r_0;
            mk[0] = ((tk_1 != 0) ? kb[2] : mk[0]);
            mi[0] = ((tk_1 != 0) ? ib[2] : mi[0]);
            unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 1);
            unsigned int pb = _shfl_xor_0;
            int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 1);
            int pc = _shfl_xor_1;
            int r_2 = (int)(pb > mk[0] || pb == mk[0] && pc < mi[0]);
            int po = r_2;
            int lower = (int)((lane_0 & 1) == 0);
            int desc = (int)((lane_0 & 2) == 0);
            int wmax = desc * lower + (1 - desc) * (1 - lower);
            int tk_3 = ((wmax != 0) ? po : 1 - po);
            mk[0] = ((tk_3 != 0) ? pb : mk[0]);
            mi[0] = ((tk_3 != 0) ? pc : mi[0]);
            unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 2);
            unsigned int pb_4 = _shfl_xor_2;
            int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 2);
            int pc_5 = _shfl_xor_3;
            int r_6 = (int)(pb_4 > mk[0] || pb_4 == mk[0] && pc_5 < mi[0]);
            int po_7 = r_6;
            int lower_8 = (int)((lane_0 & 2) == 0);
            int desc_9 = (int)((lane_0 & 4) == 0);
            int wmax_10 = desc_9 * lower_8 + (1 - desc_9) * (1 - lower_8);
            int tk_11 = ((wmax_10 != 0) ? po_7 : 1 - po_7);
            mk[0] = ((tk_11 != 0) ? pb_4 : mk[0]);
            mi[0] = ((tk_11 != 0) ? pc_5 : mi[0]);
            unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 1);
            unsigned int pb_12 = _shfl_xor_4;
            int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 1);
            int pc_13 = _shfl_xor_5;
            int r_14 = (int)(pb_12 > mk[0] || pb_12 == mk[0] && pc_13 < mi[0]);
            int po_15 = r_14;
            int lower_16 = (int)((lane_0 & 1) == 0);
            int desc_17 = (int)((lane_0 & 4) == 0);
            int wmax_18 = desc_17 * lower_16 + (1 - desc_17) * (1 - lower_16);
            int tk_19 = ((wmax_18 != 0) ? po_15 : 1 - po_15);
            mk[0] = ((tk_19 != 0) ? pb_12 : mk[0]);
            mi[0] = ((tk_19 != 0) ? pc_13 : mi[0]);
            unsigned int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 4);
            unsigned int pb_20 = _shfl_xor_6;
            int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 4);
            int pc_21 = _shfl_xor_7;
            int r_22 = (int)(pb_20 > mk[0] || pb_20 == mk[0] && pc_21 < mi[0]);
            int po_23 = r_22;
            int lower_24 = (int)((lane_0 & 4) == 0);
            int desc_25 = (int)((lane_0 & 8) == 0);
            int wmax_26 = desc_25 * lower_24 + (1 - desc_25) * (1 - lower_24);
            int tk_27 = ((wmax_26 != 0) ? po_23 : 1 - po_23);
            mk[0] = ((tk_27 != 0) ? pb_20 : mk[0]);
            mi[0] = ((tk_27 != 0) ? pc_21 : mi[0]);
            unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 2);
            unsigned int pb_28 = _shfl_xor_8;
            int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 2);
            int pc_29 = _shfl_xor_9;
            int r_30 = (int)(pb_28 > mk[0] || pb_28 == mk[0] && pc_29 < mi[0]);
            int po_31 = r_30;
            int lower_32 = (int)((lane_0 & 2) == 0);
            int desc_33 = (int)((lane_0 & 8) == 0);
            int wmax_34 = desc_33 * lower_32 + (1 - desc_33) * (1 - lower_32);
            int tk_35 = ((wmax_34 != 0) ? po_31 : 1 - po_31);
            mk[0] = ((tk_35 != 0) ? pb_28 : mk[0]);
            mi[0] = ((tk_35 != 0) ? pc_29 : mi[0]);
            unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 1);
            unsigned int pb_36 = _shfl_xor_10;
            int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 1);
            int pc_37 = _shfl_xor_11;
            int r_38 = (int)(pb_36 > mk[0] || pb_36 == mk[0] && pc_37 < mi[0]);
            int po_39 = r_38;
            int lower_40 = (int)((lane_0 & 1) == 0);
            int desc_41 = (int)((lane_0 & 8) == 0);
            int wmax_42 = desc_41 * lower_40 + (1 - desc_41) * (1 - lower_40);
            int tk_43 = ((wmax_42 != 0) ? po_39 : 1 - po_39);
            mk[0] = ((tk_43 != 0) ? pb_36 : mk[0]);
            mi[0] = ((tk_43 != 0) ? pc_37 : mi[0]);
            unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 8);
            unsigned int pb_44 = _shfl_xor_12;
            int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 8);
            int pc_45 = _shfl_xor_13;
            int r_46 = (int)(pb_44 > mk[0] || pb_44 == mk[0] && pc_45 < mi[0]);
            int po_47 = r_46;
            int lower_48 = (int)((lane_0 & 8) == 0);
            int desc_49 = (int)((lane_0 & 16) == 0);
            int wmax_50 = desc_49 * lower_48 + (1 - desc_49) * (1 - lower_48);
            int tk_51 = ((wmax_50 != 0) ? po_47 : 1 - po_47);
            mk[0] = ((tk_51 != 0) ? pb_44 : mk[0]);
            mi[0] = ((tk_51 != 0) ? pc_45 : mi[0]);
            unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 4);
            unsigned int pb_52 = _shfl_xor_14;
            int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 4);
            int pc_53 = _shfl_xor_15;
            int r_54 = (int)(pb_52 > mk[0] || pb_52 == mk[0] && pc_53 < mi[0]);
            int po_55 = r_54;
            int lower_56 = (int)((lane_0 & 4) == 0);
            int desc_57 = (int)((lane_0 & 16) == 0);
            int wmax_58 = desc_57 * lower_56 + (1 - desc_57) * (1 - lower_56);
            int tk_59 = ((wmax_58 != 0) ? po_55 : 1 - po_55);
            mk[0] = ((tk_59 != 0) ? pb_52 : mk[0]);
            mi[0] = ((tk_59 != 0) ? pc_53 : mi[0]);
            unsigned int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 2);
            unsigned int pb_60 = _shfl_xor_16;
            int _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 2);
            int pc_61 = _shfl_xor_17;
            int r_62 = (int)(pb_60 > mk[0] || pb_60 == mk[0] && pc_61 < mi[0]);
            int po_63 = r_62;
            int lower_64 = (int)((lane_0 & 2) == 0);
            int desc_65 = (int)((lane_0 & 16) == 0);
            int wmax_66 = desc_65 * lower_64 + (1 - desc_65) * (1 - lower_64);
            int tk_67 = ((wmax_66 != 0) ? po_63 : 1 - po_63);
            mk[0] = ((tk_67 != 0) ? pb_60 : mk[0]);
            mi[0] = ((tk_67 != 0) ? pc_61 : mi[0]);
            unsigned int _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 1);
            unsigned int pb_68 = _shfl_xor_18;
            int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 1);
            int pc_69 = _shfl_xor_19;
            int r_70 = (int)(pb_68 > mk[0] || pb_68 == mk[0] && pc_69 < mi[0]);
            int po_71 = r_70;
            int lower_72 = (int)((lane_0 & 1) == 0);
            int desc_73 = (int)((lane_0 & 16) == 0);
            int wmax_74 = desc_73 * lower_72 + (1 - desc_73) * (1 - lower_72);
            int tk_75 = ((wmax_74 != 0) ? po_71 : 1 - po_71);
            mk[0] = ((tk_75 != 0) ? pb_68 : mk[0]);
            mi[0] = ((tk_75 != 0) ? pc_69 : mi[0]);
            unsigned int _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 16);
            unsigned int pb_76 = _shfl_xor_20;
            int _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 16);
            int pc_77 = _shfl_xor_21;
            int r_78 = (int)(pb_76 > mk[0] || pb_76 == mk[0] && pc_77 < mi[0]);
            int po_79 = r_78;
            int lower_80 = (int)((lane_0 & 16) == 0);
            int desc_81 = (int)((lane_0 & 0) == 0);
            int wmax_82 = desc_81 * lower_80 + (1 - desc_81) * (1 - lower_80);
            int tk_83 = ((wmax_82 != 0) ? po_79 : 1 - po_79);
            mk[0] = ((tk_83 != 0) ? pb_76 : mk[0]);
            mi[0] = ((tk_83 != 0) ? pc_77 : mi[0]);
            unsigned int _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 8);
            unsigned int pb_84 = _shfl_xor_22;
            int _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 8);
            int pc_85 = _shfl_xor_23;
            int r_86 = (int)(pb_84 > mk[0] || pb_84 == mk[0] && pc_85 < mi[0]);
            int po_87 = r_86;
            int lower_88 = (int)((lane_0 & 8) == 0);
            int desc_89 = (int)((lane_0 & 0) == 0);
            int wmax_90 = desc_89 * lower_88 + (1 - desc_89) * (1 - lower_88);
            int tk_91 = ((wmax_90 != 0) ? po_87 : 1 - po_87);
            mk[0] = ((tk_91 != 0) ? pb_84 : mk[0]);
            mi[0] = ((tk_91 != 0) ? pc_85 : mi[0]);
            unsigned int _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 4);
            unsigned int pb_92 = _shfl_xor_24;
            int _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 4);
            int pc_93 = _shfl_xor_25;
            int r_94 = (int)(pb_92 > mk[0] || pb_92 == mk[0] && pc_93 < mi[0]);
            int po_95 = r_94;
            int lower_96 = (int)((lane_0 & 4) == 0);
            int desc_97 = (int)((lane_0 & 0) == 0);
            int wmax_98 = desc_97 * lower_96 + (1 - desc_97) * (1 - lower_96);
            int tk_99 = ((wmax_98 != 0) ? po_95 : 1 - po_95);
            mk[0] = ((tk_99 != 0) ? pb_92 : mk[0]);
            mi[0] = ((tk_99 != 0) ? pc_93 : mi[0]);
            unsigned int _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 2);
            unsigned int pb_100 = _shfl_xor_26;
            int _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 2);
            int pc_101 = _shfl_xor_27;
            int r_102 = (int)(pb_100 > mk[0] || pb_100 == mk[0] && pc_101 < mi[0]);
            int po_103 = r_102;
            int lower_104 = (int)((lane_0 & 2) == 0);
            int desc_105 = (int)((lane_0 & 0) == 0);
            int wmax_106 = desc_105 * lower_104 + (1 - desc_105) * (1 - lower_104);
            int tk_107 = ((wmax_106 != 0) ? po_103 : 1 - po_103);
            mk[0] = ((tk_107 != 0) ? pb_100 : mk[0]);
            mi[0] = ((tk_107 != 0) ? pc_101 : mi[0]);
            unsigned int _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, mk[0], 1);
            unsigned int pb_108 = _shfl_xor_28;
            int _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, mi[0], 1);
            int pc_109 = _shfl_xor_29;
            int r_110 = (int)(pb_108 > mk[0] || pb_108 == mk[0] && pc_109 < mi[0]);
            int po_111 = r_110;
            int lower_112 = (int)((lane_0 & 1) == 0);
            int desc_113 = (int)((lane_0 & 0) == 0);
            int wmax_114 = desc_113 * lower_112 + (1 - desc_113) * (1 - lower_112);
            int tk_115 = ((wmax_114 != 0) ? po_111 : 1 - po_111);
            mk[0] = ((tk_115 != 0) ? pb_108 : mk[0]);
            mi[0] = ((tk_115 != 0) ? pc_109 : mi[0]);
            unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, mk[0], 15);
            unsigned int bk = _shfl_0;
            int _shfl_1 = __shfl_sync(0xFFFFFFFF, mi[0], 15);
            int bi = _shfl_1;
            if (lane_0 == 0) {
                bnd_k[sub] = bk;
                bnd_i[sub] = bi;
            }
            __syncwarp();
            unsigned int yk = bnd_k[0];
            int yi = bnd_i[0];
            int r_116 = (int)(yk > bk || yk == bk && yi < bi);
            int ty = r_116;
            bk = ((ty != 0) ? yk : bk);
            bi = ((ty != 0) ? yi : bi);
            int is_real = (int)(bk != 8388607 || bi != 2147483647);
            int be = ((is_real != 0) ? bi + is_real : 2147483647);
            int sv[3];
            int cnt = 0;
            int r_117 = (int)(kb[0] > bk || kb[0] == bk && ib[0] < be);
            sv[0] = r_117;
            cnt = cnt + sv[0];
            int r_118 = (int)(kb[1] > bk || kb[1] == bk && ib[1] < be);
            sv[1] = r_118;
            cnt = cnt + sv[1];
            int r_119 = (int)(kb[2] > bk || kb[2] == bk && ib[2] < be);
            sv[2] = r_119;
            cnt = cnt + sv[2];
            uint32_t _warp_scan_sum_u32_0 = (unsigned int)cnt;
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(1));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(2));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(4));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(8));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(16));
            unsigned int inc = _warp_scan_sum_u32_0;
            if (lane_0 == 31) {
                cnt_w[sub] = (int)inc;
            }
            __syncwarp();
            int base = (int)inc - cnt;
            int total = 0;
            int c2 = cnt_w[0];
            base = base + ((sub > 0) ? c2 : 0);
            total = total + c2;
            int idx = 0;
            if (sv[0] != 0 && base + idx < 32) {
                surv_k[base + idx] = kb[0];
                surv_i[base + idx] = ib[0];
            }
            idx = idx + sv[0];
            if (sv[1] != 0 && base + idx < 32) {
                surv_k[base + idx] = kb[1];
                surv_i[base + idx] = ib[1];
            }
            idx = idx + sv[1];
            if (sv[2] != 0 && base + idx < 32) {
                surv_k[base + idx] = kb[2];
                surv_i[base + idx] = ib[2];
            }
            idx = idx + sv[2];
            __syncwarp();
            S = total;
            int s_prev = 0;
            while (S > 32) {
                hk[0] = surv_k[tr];
                hi[0] = surv_i[tr];
                unsigned int _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 1);
                unsigned int pb_0 = _shfl_xor_30;
                int _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 1);
                int pc_1 = _shfl_xor_31;
                int r_3 = (int)(pb_0 > hk[0] || pb_0 == hk[0] && pc_1 < hi[0]);
                int po_4 = r_3;
                int lower_5 = (int)((lane_0 & 1) == 0);
                int desc_6 = (int)((lane_0 & 2) == 0);
                int wmax_7 = desc_6 * lower_5 + (1 - desc_6) * (1 - lower_5);
                int tk_8 = ((wmax_7 != 0) ? po_4 : 1 - po_4);
                hk[0] = ((tk_8 != 0) ? pb_0 : hk[0]);
                hi[0] = ((tk_8 != 0) ? pc_1 : hi[0]);
                unsigned int _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 2);
                unsigned int pb_9 = _shfl_xor_32;
                int _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 2);
                int pc_10 = _shfl_xor_33;
                int r_11 = (int)(pb_9 > hk[0] || pb_9 == hk[0] && pc_10 < hi[0]);
                int po_12 = r_11;
                int lower_13 = (int)((lane_0 & 2) == 0);
                int desc_14 = (int)((lane_0 & 4) == 0);
                int wmax_15 = desc_14 * lower_13 + (1 - desc_14) * (1 - lower_13);
                int tk_16 = ((wmax_15 != 0) ? po_12 : 1 - po_12);
                hk[0] = ((tk_16 != 0) ? pb_9 : hk[0]);
                hi[0] = ((tk_16 != 0) ? pc_10 : hi[0]);
                unsigned int _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 1);
                unsigned int pb_17 = _shfl_xor_34;
                int _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 1);
                int pc_18 = _shfl_xor_35;
                int r_19 = (int)(pb_17 > hk[0] || pb_17 == hk[0] && pc_18 < hi[0]);
                int po_20 = r_19;
                int lower_21 = (int)((lane_0 & 1) == 0);
                int desc_22 = (int)((lane_0 & 4) == 0);
                int wmax_23 = desc_22 * lower_21 + (1 - desc_22) * (1 - lower_21);
                int tk_24 = ((wmax_23 != 0) ? po_20 : 1 - po_20);
                hk[0] = ((tk_24 != 0) ? pb_17 : hk[0]);
                hi[0] = ((tk_24 != 0) ? pc_18 : hi[0]);
                unsigned int _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 4);
                unsigned int pb_25 = _shfl_xor_36;
                int _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 4);
                int pc_26 = _shfl_xor_37;
                int r_27 = (int)(pb_25 > hk[0] || pb_25 == hk[0] && pc_26 < hi[0]);
                int po_28 = r_27;
                int lower_29 = (int)((lane_0 & 4) == 0);
                int desc_30 = (int)((lane_0 & 8) == 0);
                int wmax_31 = desc_30 * lower_29 + (1 - desc_30) * (1 - lower_29);
                int tk_32 = ((wmax_31 != 0) ? po_28 : 1 - po_28);
                hk[0] = ((tk_32 != 0) ? pb_25 : hk[0]);
                hi[0] = ((tk_32 != 0) ? pc_26 : hi[0]);
                unsigned int _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 2);
                unsigned int pb_33 = _shfl_xor_38;
                int _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 2);
                int pc_34 = _shfl_xor_39;
                int r_35 = (int)(pb_33 > hk[0] || pb_33 == hk[0] && pc_34 < hi[0]);
                int po_36 = r_35;
                int lower_37 = (int)((lane_0 & 2) == 0);
                int desc_38 = (int)((lane_0 & 8) == 0);
                int wmax_39 = desc_38 * lower_37 + (1 - desc_38) * (1 - lower_37);
                int tk_40 = ((wmax_39 != 0) ? po_36 : 1 - po_36);
                hk[0] = ((tk_40 != 0) ? pb_33 : hk[0]);
                hi[0] = ((tk_40 != 0) ? pc_34 : hi[0]);
                unsigned int _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 1);
                unsigned int pb_41 = _shfl_xor_40;
                int _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 1);
                int pc_42 = _shfl_xor_41;
                int r_43 = (int)(pb_41 > hk[0] || pb_41 == hk[0] && pc_42 < hi[0]);
                int po_44 = r_43;
                int lower_45 = (int)((lane_0 & 1) == 0);
                int desc_46 = (int)((lane_0 & 8) == 0);
                int wmax_47 = desc_46 * lower_45 + (1 - desc_46) * (1 - lower_45);
                int tk_48 = ((wmax_47 != 0) ? po_44 : 1 - po_44);
                hk[0] = ((tk_48 != 0) ? pb_41 : hk[0]);
                hi[0] = ((tk_48 != 0) ? pc_42 : hi[0]);
                unsigned int _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 8);
                unsigned int pb_49 = _shfl_xor_42;
                int _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 8);
                int pc_50 = _shfl_xor_43;
                int r_51 = (int)(pb_49 > hk[0] || pb_49 == hk[0] && pc_50 < hi[0]);
                int po_52 = r_51;
                int lower_53 = (int)((lane_0 & 8) == 0);
                int desc_54 = (int)((lane_0 & 16) == 0);
                int wmax_55 = desc_54 * lower_53 + (1 - desc_54) * (1 - lower_53);
                int tk_56 = ((wmax_55 != 0) ? po_52 : 1 - po_52);
                hk[0] = ((tk_56 != 0) ? pb_49 : hk[0]);
                hi[0] = ((tk_56 != 0) ? pc_50 : hi[0]);
                unsigned int _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 4);
                unsigned int pb_57 = _shfl_xor_44;
                int _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 4);
                int pc_58 = _shfl_xor_45;
                int r_59 = (int)(pb_57 > hk[0] || pb_57 == hk[0] && pc_58 < hi[0]);
                int po_60 = r_59;
                int lower_61 = (int)((lane_0 & 4) == 0);
                int desc_62 = (int)((lane_0 & 16) == 0);
                int wmax_63 = desc_62 * lower_61 + (1 - desc_62) * (1 - lower_61);
                int tk_64 = ((wmax_63 != 0) ? po_60 : 1 - po_60);
                hk[0] = ((tk_64 != 0) ? pb_57 : hk[0]);
                hi[0] = ((tk_64 != 0) ? pc_58 : hi[0]);
                unsigned int _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 2);
                unsigned int pb_65 = _shfl_xor_46;
                int _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 2);
                int pc_66 = _shfl_xor_47;
                int r_67 = (int)(pb_65 > hk[0] || pb_65 == hk[0] && pc_66 < hi[0]);
                int po_68 = r_67;
                int lower_69 = (int)((lane_0 & 2) == 0);
                int desc_70 = (int)((lane_0 & 16) == 0);
                int wmax_71 = desc_70 * lower_69 + (1 - desc_70) * (1 - lower_69);
                int tk_72 = ((wmax_71 != 0) ? po_68 : 1 - po_68);
                hk[0] = ((tk_72 != 0) ? pb_65 : hk[0]);
                hi[0] = ((tk_72 != 0) ? pc_66 : hi[0]);
                unsigned int _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 1);
                unsigned int pb_73 = _shfl_xor_48;
                int _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 1);
                int pc_74 = _shfl_xor_49;
                int r_75 = (int)(pb_73 > hk[0] || pb_73 == hk[0] && pc_74 < hi[0]);
                int po_76 = r_75;
                int lower_77 = (int)((lane_0 & 1) == 0);
                int desc_78 = (int)((lane_0 & 16) == 0);
                int wmax_79 = desc_78 * lower_77 + (1 - desc_78) * (1 - lower_77);
                int tk_80 = ((wmax_79 != 0) ? po_76 : 1 - po_76);
                hk[0] = ((tk_80 != 0) ? pb_73 : hk[0]);
                hi[0] = ((tk_80 != 0) ? pc_74 : hi[0]);
                unsigned int _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 16);
                unsigned int pb_81 = _shfl_xor_50;
                int _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 16);
                int pc_82 = _shfl_xor_51;
                int r_83 = (int)(pb_81 > hk[0] || pb_81 == hk[0] && pc_82 < hi[0]);
                int po_84 = r_83;
                int lower_85 = (int)((lane_0 & 16) == 0);
                int desc_86 = (int)((lane_0 & 0) == 0);
                int wmax_87 = desc_86 * lower_85 + (1 - desc_86) * (1 - lower_85);
                int tk_88 = ((wmax_87 != 0) ? po_84 : 1 - po_84);
                hk[0] = ((tk_88 != 0) ? pb_81 : hk[0]);
                hi[0] = ((tk_88 != 0) ? pc_82 : hi[0]);
                unsigned int _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 8);
                unsigned int pb_89 = _shfl_xor_52;
                int _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 8);
                int pc_90 = _shfl_xor_53;
                int r_91 = (int)(pb_89 > hk[0] || pb_89 == hk[0] && pc_90 < hi[0]);
                int po_92 = r_91;
                int lower_93 = (int)((lane_0 & 8) == 0);
                int desc_94 = (int)((lane_0 & 0) == 0);
                int wmax_95 = desc_94 * lower_93 + (1 - desc_94) * (1 - lower_93);
                int tk_96 = ((wmax_95 != 0) ? po_92 : 1 - po_92);
                hk[0] = ((tk_96 != 0) ? pb_89 : hk[0]);
                hi[0] = ((tk_96 != 0) ? pc_90 : hi[0]);
                unsigned int _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 4);
                unsigned int pb_97 = _shfl_xor_54;
                int _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 4);
                int pc_98 = _shfl_xor_55;
                int r_99 = (int)(pb_97 > hk[0] || pb_97 == hk[0] && pc_98 < hi[0]);
                int po_100 = r_99;
                int lower_101 = (int)((lane_0 & 4) == 0);
                int desc_102 = (int)((lane_0 & 0) == 0);
                int wmax_103 = desc_102 * lower_101 + (1 - desc_102) * (1 - lower_101);
                int tk_104 = ((wmax_103 != 0) ? po_100 : 1 - po_100);
                hk[0] = ((tk_104 != 0) ? pb_97 : hk[0]);
                hi[0] = ((tk_104 != 0) ? pc_98 : hi[0]);
                unsigned int _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 2);
                unsigned int pb_105 = _shfl_xor_56;
                int _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 2);
                int pc_106 = _shfl_xor_57;
                int r_107 = (int)(pb_105 > hk[0] || pb_105 == hk[0] && pc_106 < hi[0]);
                int po_108 = r_107;
                int lower_109 = (int)((lane_0 & 2) == 0);
                int desc_110 = (int)((lane_0 & 0) == 0);
                int wmax_111 = desc_110 * lower_109 + (1 - desc_110) * (1 - lower_109);
                int tk_112 = ((wmax_111 != 0) ? po_108 : 1 - po_108);
                hk[0] = ((tk_112 != 0) ? pb_105 : hk[0]);
                hi[0] = ((tk_112 != 0) ? pc_106 : hi[0]);
                unsigned int _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 1);
                unsigned int pb_113 = _shfl_xor_58;
                int _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 1);
                int pc_114 = _shfl_xor_59;
                int r_115 = (int)(pb_113 > hk[0] || pb_113 == hk[0] && pc_114 < hi[0]);
                int po_116 = r_115;
                int lower_117 = (int)((lane_0 & 1) == 0);
                int desc_118 = (int)((lane_0 & 0) == 0);
                int wmax_119 = desc_118 * lower_117 + (1 - desc_118) * (1 - lower_117);
                int tk_120 = ((wmax_119 != 0) ? po_116 : 1 - po_116);
                hk[0] = ((tk_120 != 0) ? pb_113 : hk[0]);
                hi[0] = ((tk_120 != 0) ? pc_114 : hi[0]);
                sort_k[sub * 32 + lane_0] = hk[0];
                sort_i[sub * 32 + lane_0] = hi[0];
                __syncwarp();
                int rs_hi = (int)(lane_0 >= 16);
                unsigned int _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 16);
                unsigned int rs_xk = _shfl_xor_60;
                int _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 16);
                int rs_xi = _shfl_xor_61;
                unsigned int rs_ck = ((rs_hi != 0) ? rs_xk : hk[0]);
                int rs_ci = ((rs_hi != 0) ? rs_xi : hi[0]);
                int rs_part = 0;
                int rs_w = rs_hi;
                int rs_pos = 0;
                unsigned int rs_pk = sort_k[rs_w * 32 + rs_pos + 15];
                int rs_pi = sort_i[rs_w * 32 + rs_pos + 15];
                int r_121 = (int)(rs_pk > rs_ck || rs_pk == rs_ck && rs_pi < rs_ci);
                int rs_adv = r_121;
                rs_pos = rs_pos + rs_adv * 16;
                unsigned int rs_pk_122 = sort_k[rs_w * 32 + rs_pos + 7];
                int rs_pi_123 = sort_i[rs_w * 32 + rs_pos + 7];
                int r_124 = (int)(rs_pk_122 > rs_ck || rs_pk_122 == rs_ck && rs_pi_123 < rs_ci);
                int rs_adv_125 = r_124;
                rs_pos = rs_pos + rs_adv_125 * 8;
                unsigned int rs_pk_126 = sort_k[rs_w * 32 + rs_pos + 3];
                int rs_pi_127 = sort_i[rs_w * 32 + rs_pos + 3];
                int r_128 = (int)(rs_pk_126 > rs_ck || rs_pk_126 == rs_ck && rs_pi_127 < rs_ci);
                int rs_adv_129 = r_128;
                rs_pos = rs_pos + rs_adv_129 * 4;
                unsigned int rs_pk_130 = sort_k[rs_w * 32 + rs_pos + 1];
                int rs_pi_131 = sort_i[rs_w * 32 + rs_pos + 1];
                int r_132 = (int)(rs_pk_130 > rs_ck || rs_pk_130 == rs_ck && rs_pi_131 < rs_ci);
                int rs_adv_133 = r_132;
                rs_pos = rs_pos + rs_adv_133 * 2;
                unsigned int rs_pk_134 = sort_k[rs_w * 32 + rs_pos];
                int rs_pi_135 = sort_i[rs_w * 32 + rs_pos];
                int r_136 = (int)(rs_pk_134 > rs_ck || rs_pk_134 == rs_ck && rs_pi_135 < rs_ci);
                int rs_adv_137 = r_136;
                rs_pos = rs_pos + rs_adv_137;
                int rs_ok = (int)(rs_w != sub && rs_w < 1);
                rs_part = rs_part + rs_pos * rs_ok;
                int _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, rs_part, 16);
                int rs_other = _shfl_xor_62;
                int rs_rank = lane_0 + rs_part + rs_other;
                rnk = rs_rank;
                if (rnk == 15) {
                    bnd_k[0] = hk[0];
                    bnd_i[0] = hi[0];
                }
                __syncwarp();
                bk = bnd_k[0];
                bi = bnd_i[0];
                be = bi + 1;
                s_prev = S;
                int cnt_138 = 0;
                int r_139 = (int)(kb[0] > bk || kb[0] == bk && ib[0] < be);
                sv[0] = r_139;
                cnt_138 = cnt_138 + sv[0];
                int r_140 = (int)(kb[1] > bk || kb[1] == bk && ib[1] < be);
                sv[1] = r_140;
                cnt_138 = cnt_138 + sv[1];
                int r_141 = (int)(kb[2] > bk || kb[2] == bk && ib[2] < be);
                sv[2] = r_141;
                cnt_138 = cnt_138 + sv[2];
                uint32_t _warp_scan_sum_u32_1 = (unsigned int)cnt_138;
                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(1));
                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(2));
                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(4));
                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(8));
                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(16));
                unsigned int inc_142 = _warp_scan_sum_u32_1;
                if (lane_0 == 31) {
                    cnt_w[sub] = (int)inc_142;
                }
                __syncwarp();
                int base_143 = (int)inc_142 - cnt_138;
                int total_144 = 0;
                int c2_145 = cnt_w[0];
                base_143 = base_143 + ((sub > 0) ? c2_145 : 0);
                total_144 = total_144 + c2_145;
                int idx_146 = 0;
                if (sv[0] != 0 && base_143 + idx_146 < 32) {
                    surv_k[base_143 + idx_146] = kb[0];
                    surv_i[base_143 + idx_146] = ib[0];
                }
                idx_146 = idx_146 + sv[0];
                if (sv[1] != 0 && base_143 + idx_146 < 32) {
                    surv_k[base_143 + idx_146] = kb[1];
                    surv_i[base_143 + idx_146] = ib[1];
                }
                idx_146 = idx_146 + sv[1];
                if (sv[2] != 0 && base_143 + idx_146 < 32) {
                    surv_k[base_143 + idx_146] = kb[2];
                    surv_i[base_143 + idx_146] = ib[2];
                }
                idx_146 = idx_146 + sv[2];
                __syncwarp();
                S = total_144;
                if (S >= s_prev) {
                    if (tr == 0) {
                        status[0] = 4;
                    }
                    S = 0;
                }
            }
            hk[0] = ((tr < S) ? surv_k[tr] : 8388607);
            hi[0] = ((tr < S) ? surv_i[tr] : 2147483647);
            unsigned int _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 1);
            unsigned int pb_120 = _shfl_xor_63;
            int _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 1);
            int pc_121 = _shfl_xor_64;
            int r_122 = (int)(pb_120 > hk[0] || pb_120 == hk[0] && pc_121 < hi[0]);
            int po_123 = r_122;
            int lower_124 = (int)((lane_0 & 1) == 0);
            int desc_125 = (int)((lane_0 & 2) == 0);
            int wmax_126 = desc_125 * lower_124 + (1 - desc_125) * (1 - lower_124);
            int tk_127 = ((wmax_126 != 0) ? po_123 : 1 - po_123);
            hk[0] = ((tk_127 != 0) ? pb_120 : hk[0]);
            hi[0] = ((tk_127 != 0) ? pc_121 : hi[0]);
            unsigned int _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 2);
            unsigned int pb_128 = _shfl_xor_65;
            int _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 2);
            int pc_129 = _shfl_xor_66;
            int r_130 = (int)(pb_128 > hk[0] || pb_128 == hk[0] && pc_129 < hi[0]);
            int po_131 = r_130;
            int lower_132 = (int)((lane_0 & 2) == 0);
            int desc_133 = (int)((lane_0 & 4) == 0);
            int wmax_134 = desc_133 * lower_132 + (1 - desc_133) * (1 - lower_132);
            int tk_135 = ((wmax_134 != 0) ? po_131 : 1 - po_131);
            hk[0] = ((tk_135 != 0) ? pb_128 : hk[0]);
            hi[0] = ((tk_135 != 0) ? pc_129 : hi[0]);
            unsigned int _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 1);
            unsigned int pb_136 = _shfl_xor_67;
            int _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 1);
            int pc_137 = _shfl_xor_68;
            int r_138 = (int)(pb_136 > hk[0] || pb_136 == hk[0] && pc_137 < hi[0]);
            int po_139 = r_138;
            int lower_140 = (int)((lane_0 & 1) == 0);
            int desc_141 = (int)((lane_0 & 4) == 0);
            int wmax_142 = desc_141 * lower_140 + (1 - desc_141) * (1 - lower_140);
            int tk_143 = ((wmax_142 != 0) ? po_139 : 1 - po_139);
            hk[0] = ((tk_143 != 0) ? pb_136 : hk[0]);
            hi[0] = ((tk_143 != 0) ? pc_137 : hi[0]);
            unsigned int _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 4);
            unsigned int pb_144 = _shfl_xor_69;
            int _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 4);
            int pc_145 = _shfl_xor_70;
            int r_146 = (int)(pb_144 > hk[0] || pb_144 == hk[0] && pc_145 < hi[0]);
            int po_147 = r_146;
            int lower_148 = (int)((lane_0 & 4) == 0);
            int desc_149 = (int)((lane_0 & 8) == 0);
            int wmax_150 = desc_149 * lower_148 + (1 - desc_149) * (1 - lower_148);
            int tk_151 = ((wmax_150 != 0) ? po_147 : 1 - po_147);
            hk[0] = ((tk_151 != 0) ? pb_144 : hk[0]);
            hi[0] = ((tk_151 != 0) ? pc_145 : hi[0]);
            unsigned int _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 2);
            unsigned int pb_152 = _shfl_xor_71;
            int _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 2);
            int pc_153 = _shfl_xor_72;
            int r_154 = (int)(pb_152 > hk[0] || pb_152 == hk[0] && pc_153 < hi[0]);
            int po_155 = r_154;
            int lower_156 = (int)((lane_0 & 2) == 0);
            int desc_157 = (int)((lane_0 & 8) == 0);
            int wmax_158 = desc_157 * lower_156 + (1 - desc_157) * (1 - lower_156);
            int tk_159 = ((wmax_158 != 0) ? po_155 : 1 - po_155);
            hk[0] = ((tk_159 != 0) ? pb_152 : hk[0]);
            hi[0] = ((tk_159 != 0) ? pc_153 : hi[0]);
            unsigned int _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 1);
            unsigned int pb_160 = _shfl_xor_73;
            int _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 1);
            int pc_161 = _shfl_xor_74;
            int r_162 = (int)(pb_160 > hk[0] || pb_160 == hk[0] && pc_161 < hi[0]);
            int po_163 = r_162;
            int lower_164 = (int)((lane_0 & 1) == 0);
            int desc_165 = (int)((lane_0 & 8) == 0);
            int wmax_166 = desc_165 * lower_164 + (1 - desc_165) * (1 - lower_164);
            int tk_167 = ((wmax_166 != 0) ? po_163 : 1 - po_163);
            hk[0] = ((tk_167 != 0) ? pb_160 : hk[0]);
            hi[0] = ((tk_167 != 0) ? pc_161 : hi[0]);
            unsigned int _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 8);
            unsigned int pb_168 = _shfl_xor_75;
            int _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 8);
            int pc_169 = _shfl_xor_76;
            int r_170 = (int)(pb_168 > hk[0] || pb_168 == hk[0] && pc_169 < hi[0]);
            int po_171 = r_170;
            int lower_172 = (int)((lane_0 & 8) == 0);
            int desc_173 = (int)((lane_0 & 16) == 0);
            int wmax_174 = desc_173 * lower_172 + (1 - desc_173) * (1 - lower_172);
            int tk_175 = ((wmax_174 != 0) ? po_171 : 1 - po_171);
            hk[0] = ((tk_175 != 0) ? pb_168 : hk[0]);
            hi[0] = ((tk_175 != 0) ? pc_169 : hi[0]);
            unsigned int _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 4);
            unsigned int pb_176 = _shfl_xor_77;
            int _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 4);
            int pc_177 = _shfl_xor_78;
            int r_178 = (int)(pb_176 > hk[0] || pb_176 == hk[0] && pc_177 < hi[0]);
            int po_179 = r_178;
            int lower_180 = (int)((lane_0 & 4) == 0);
            int desc_181 = (int)((lane_0 & 16) == 0);
            int wmax_182 = desc_181 * lower_180 + (1 - desc_181) * (1 - lower_180);
            int tk_183 = ((wmax_182 != 0) ? po_179 : 1 - po_179);
            hk[0] = ((tk_183 != 0) ? pb_176 : hk[0]);
            hi[0] = ((tk_183 != 0) ? pc_177 : hi[0]);
            unsigned int _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 2);
            unsigned int pb_184 = _shfl_xor_79;
            int _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 2);
            int pc_185 = _shfl_xor_80;
            int r_186 = (int)(pb_184 > hk[0] || pb_184 == hk[0] && pc_185 < hi[0]);
            int po_187 = r_186;
            int lower_188 = (int)((lane_0 & 2) == 0);
            int desc_189 = (int)((lane_0 & 16) == 0);
            int wmax_190 = desc_189 * lower_188 + (1 - desc_189) * (1 - lower_188);
            int tk_191 = ((wmax_190 != 0) ? po_187 : 1 - po_187);
            hk[0] = ((tk_191 != 0) ? pb_184 : hk[0]);
            hi[0] = ((tk_191 != 0) ? pc_185 : hi[0]);
            unsigned int _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 1);
            unsigned int pb_192 = _shfl_xor_81;
            int _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 1);
            int pc_193 = _shfl_xor_82;
            int r_194 = (int)(pb_192 > hk[0] || pb_192 == hk[0] && pc_193 < hi[0]);
            int po_195 = r_194;
            int lower_196 = (int)((lane_0 & 1) == 0);
            int desc_197 = (int)((lane_0 & 16) == 0);
            int wmax_198 = desc_197 * lower_196 + (1 - desc_197) * (1 - lower_196);
            int tk_199 = ((wmax_198 != 0) ? po_195 : 1 - po_195);
            hk[0] = ((tk_199 != 0) ? pb_192 : hk[0]);
            hi[0] = ((tk_199 != 0) ? pc_193 : hi[0]);
            unsigned int _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 16);
            unsigned int pb_200 = _shfl_xor_83;
            int _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 16);
            int pc_201 = _shfl_xor_84;
            int r_202 = (int)(pb_200 > hk[0] || pb_200 == hk[0] && pc_201 < hi[0]);
            int po_203 = r_202;
            int lower_204 = (int)((lane_0 & 16) == 0);
            int desc_205 = (int)((lane_0 & 0) == 0);
            int wmax_206 = desc_205 * lower_204 + (1 - desc_205) * (1 - lower_204);
            int tk_207 = ((wmax_206 != 0) ? po_203 : 1 - po_203);
            hk[0] = ((tk_207 != 0) ? pb_200 : hk[0]);
            hi[0] = ((tk_207 != 0) ? pc_201 : hi[0]);
            unsigned int _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 8);
            unsigned int pb_208 = _shfl_xor_85;
            int _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 8);
            int pc_209 = _shfl_xor_86;
            int r_210 = (int)(pb_208 > hk[0] || pb_208 == hk[0] && pc_209 < hi[0]);
            int po_211 = r_210;
            int lower_212 = (int)((lane_0 & 8) == 0);
            int desc_213 = (int)((lane_0 & 0) == 0);
            int wmax_214 = desc_213 * lower_212 + (1 - desc_213) * (1 - lower_212);
            int tk_215 = ((wmax_214 != 0) ? po_211 : 1 - po_211);
            hk[0] = ((tk_215 != 0) ? pb_208 : hk[0]);
            hi[0] = ((tk_215 != 0) ? pc_209 : hi[0]);
            unsigned int _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 4);
            unsigned int pb_216 = _shfl_xor_87;
            int _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 4);
            int pc_217 = _shfl_xor_88;
            int r_218 = (int)(pb_216 > hk[0] || pb_216 == hk[0] && pc_217 < hi[0]);
            int po_219 = r_218;
            int lower_220 = (int)((lane_0 & 4) == 0);
            int desc_221 = (int)((lane_0 & 0) == 0);
            int wmax_222 = desc_221 * lower_220 + (1 - desc_221) * (1 - lower_220);
            int tk_223 = ((wmax_222 != 0) ? po_219 : 1 - po_219);
            hk[0] = ((tk_223 != 0) ? pb_216 : hk[0]);
            hi[0] = ((tk_223 != 0) ? pc_217 : hi[0]);
            unsigned int _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 2);
            unsigned int pb_224 = _shfl_xor_89;
            int _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 2);
            int pc_225 = _shfl_xor_90;
            int r_226 = (int)(pb_224 > hk[0] || pb_224 == hk[0] && pc_225 < hi[0]);
            int po_227 = r_226;
            int lower_228 = (int)((lane_0 & 2) == 0);
            int desc_229 = (int)((lane_0 & 0) == 0);
            int wmax_230 = desc_229 * lower_228 + (1 - desc_229) * (1 - lower_228);
            int tk_231 = ((wmax_230 != 0) ? po_227 : 1 - po_227);
            hk[0] = ((tk_231 != 0) ? pb_224 : hk[0]);
            hi[0] = ((tk_231 != 0) ? pc_225 : hi[0]);
            unsigned int _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 1);
            unsigned int pb_232 = _shfl_xor_91;
            int _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 1);
            int pc_233 = _shfl_xor_92;
            int r_234 = (int)(pb_232 > hk[0] || pb_232 == hk[0] && pc_233 < hi[0]);
            int po_235 = r_234;
            int lower_236 = (int)((lane_0 & 1) == 0);
            int desc_237 = (int)((lane_0 & 0) == 0);
            int wmax_238 = desc_237 * lower_236 + (1 - desc_237) * (1 - lower_236);
            int tk_239 = ((wmax_238 != 0) ? po_235 : 1 - po_235);
            hk[0] = ((tk_239 != 0) ? pb_232 : hk[0]);
            hi[0] = ((tk_239 != 0) ? pc_233 : hi[0]);
            sort_k[sub * 32 + lane_0] = hk[0];
            sort_i[sub * 32 + lane_0] = hi[0];
            __syncwarp();
            int rs_hi_1 = (int)(lane_0 >= 16);
            unsigned int _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, hk[0], 16);
            unsigned int rs_xk_1 = _shfl_xor_93;
            int _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, hi[0], 16);
            int rs_xi_1 = _shfl_xor_94;
            unsigned int rs_ck_1 = ((rs_hi_1 != 0) ? rs_xk_1 : hk[0]);
            int rs_ci_1 = ((rs_hi_1 != 0) ? rs_xi_1 : hi[0]);
            int rs_part_1 = 0;
            int rs_w_1 = rs_hi_1;
            int rs_pos_1 = 0;
            unsigned int rs_pk_1 = sort_k[rs_w_1 * 32 + rs_pos_1 + 15];
            int rs_pi_1 = sort_i[rs_w_1 * 32 + rs_pos_1 + 15];
            int r_240 = (int)(rs_pk_1 > rs_ck_1 || rs_pk_1 == rs_ck_1 && rs_pi_1 < rs_ci_1);
            int rs_adv_1 = r_240;
            rs_pos_1 = rs_pos_1 + rs_adv_1 * 16;
            unsigned int rs_pk_241 = sort_k[rs_w_1 * 32 + rs_pos_1 + 7];
            int rs_pi_242 = sort_i[rs_w_1 * 32 + rs_pos_1 + 7];
            int r_243 = (int)(rs_pk_241 > rs_ck_1 || rs_pk_241 == rs_ck_1 && rs_pi_242 < rs_ci_1);
            int rs_adv_244 = r_243;
            rs_pos_1 = rs_pos_1 + rs_adv_244 * 8;
            unsigned int rs_pk_245 = sort_k[rs_w_1 * 32 + rs_pos_1 + 3];
            int rs_pi_246 = sort_i[rs_w_1 * 32 + rs_pos_1 + 3];
            int r_247 = (int)(rs_pk_245 > rs_ck_1 || rs_pk_245 == rs_ck_1 && rs_pi_246 < rs_ci_1);
            int rs_adv_248 = r_247;
            rs_pos_1 = rs_pos_1 + rs_adv_248 * 4;
            unsigned int rs_pk_249 = sort_k[rs_w_1 * 32 + rs_pos_1 + 1];
            int rs_pi_250 = sort_i[rs_w_1 * 32 + rs_pos_1 + 1];
            int r_251 = (int)(rs_pk_249 > rs_ck_1 || rs_pk_249 == rs_ck_1 && rs_pi_250 < rs_ci_1);
            int rs_adv_252 = r_251;
            rs_pos_1 = rs_pos_1 + rs_adv_252 * 2;
            unsigned int rs_pk_253 = sort_k[rs_w_1 * 32 + rs_pos_1];
            int rs_pi_254 = sort_i[rs_w_1 * 32 + rs_pos_1];
            int r_255 = (int)(rs_pk_253 > rs_ck_1 || rs_pk_253 == rs_ck_1 && rs_pi_254 < rs_ci_1);
            int rs_adv_256 = r_255;
            rs_pos_1 = rs_pos_1 + rs_adv_256;
            int rs_ok_1 = (int)(rs_w_1 != sub && rs_w_1 < 1);
            rs_part_1 = rs_part_1 + rs_pos_1 * rs_ok_1;
            int _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, rs_part_1, 16);
            int rs_other_1 = _shfl_xor_95;
            int rs_rank_1 = lane_0 + rs_part_1 + rs_other_1;
            rnk = rs_rank_1;
            h_real = (int)(hk[0] != 8388607 || hi[0] != 2147483647);
            if (n_ch > c + 1) {
                if (h_real != 0 && rnk < 16) {
                    carry_k[rnk] = hk[0];
                    carry_i[rnk] = hi[0];
                }
                if (tr >= S && tr < 16) {
                    carry_k[tr] = 8388607;
                    carry_i[tr] = 2147483647;
                }
                __syncwarp();
            }
            c = c + 1;
            more = (int)(c < n_ch);
        }
        if (fast != 0) {
            if (sub == 0) {
                unsigned int fk[2];
                int fi[2];
                unsigned int fkw = 0;
                fkw = reinterpret_cast<unsigned int*>(&wv[0])[0];
                fk[0] = ((n_e > 2 * lane_0) ? fkw : 8388607);
                fi[0] = ((n_e > 2 * lane_0) ? wv[1] : 2147483647);
                fkw = reinterpret_cast<unsigned int*>(&wv[2])[0];
                fk[1] = ((n_e > 2 * lane_0 + 1) ? fkw : 8388607);
                fi[1] = ((n_e > 2 * lane_0 + 1) ? wv[3] : 2147483647);
                unsigned int _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 1);
                unsigned int pb_1 = _shfl_xor_96;
                int _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 1);
                int pc_2 = _shfl_xor_97;
                int r_1 = (int)(pb_1 > fk[0] || pb_1 == fk[0] && pc_2 < fi[0]);
                int po_1 = r_1;
                int lower_1 = (int)((lane_0 & 1) == 0);
                int desc_1 = (int)((lane_0 & 2) == 0);
                int wmax_1 = desc_1 * lower_1 + (1 - desc_1) * (1 - lower_1);
                int tk_2 = ((wmax_1 != 0) ? po_1 : 1 - po_1);
                fk[0] = ((tk_2 != 0) ? pb_1 : fk[0]);
                fi[0] = ((tk_2 != 0) ? pc_2 : fi[0]);
                unsigned int _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 1);
                unsigned int pb_0_1 = _shfl_xor_98;
                int _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 1);
                int pc_1_1 = _shfl_xor_99;
                int r_2_1 = (int)(pb_0_1 > fk[1] || pb_0_1 == fk[1] && pc_1_1 < fi[1]);
                int po_3 = r_2_1;
                int lower_4 = (int)((lane_0 & 1) == 0);
                int desc_5 = (int)((lane_0 & 2) == 0);
                int wmax_6 = desc_5 * lower_4 + (1 - desc_5) * (1 - lower_4);
                int tk_7 = ((wmax_6 != 0) ? po_3 : 1 - po_3);
                fk[1] = ((tk_7 != 0) ? pb_0_1 : fk[1]);
                fi[1] = ((tk_7 != 0) ? pc_1_1 : fi[1]);
                unsigned int _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 2);
                unsigned int pb_8 = _shfl_xor_100;
                int _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 2);
                int pc_9 = _shfl_xor_101;
                int r_10 = (int)(pb_8 > fk[0] || pb_8 == fk[0] && pc_9 < fi[0]);
                int po_11 = r_10;
                int lower_12 = (int)((lane_0 & 2) == 0);
                int desc_13 = (int)((lane_0 & 4) == 0);
                int wmax_14 = desc_13 * lower_12 + (1 - desc_13) * (1 - lower_12);
                int tk_15 = ((wmax_14 != 0) ? po_11 : 1 - po_11);
                fk[0] = ((tk_15 != 0) ? pb_8 : fk[0]);
                fi[0] = ((tk_15 != 0) ? pc_9 : fi[0]);
                unsigned int _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 2);
                unsigned int pb_16 = _shfl_xor_102;
                int _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 2);
                int pc_17 = _shfl_xor_103;
                int r_18 = (int)(pb_16 > fk[1] || pb_16 == fk[1] && pc_17 < fi[1]);
                int po_19 = r_18;
                int lower_20 = (int)((lane_0 & 2) == 0);
                int desc_21 = (int)((lane_0 & 4) == 0);
                int wmax_22 = desc_21 * lower_20 + (1 - desc_21) * (1 - lower_20);
                int tk_23 = ((wmax_22 != 0) ? po_19 : 1 - po_19);
                fk[1] = ((tk_23 != 0) ? pb_16 : fk[1]);
                fi[1] = ((tk_23 != 0) ? pc_17 : fi[1]);
                unsigned int _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 1);
                unsigned int pb_24 = _shfl_xor_104;
                int _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 1);
                int pc_25 = _shfl_xor_105;
                int r_26 = (int)(pb_24 > fk[0] || pb_24 == fk[0] && pc_25 < fi[0]);
                int po_27 = r_26;
                int lower_28 = (int)((lane_0 & 1) == 0);
                int desc_29 = (int)((lane_0 & 4) == 0);
                int wmax_30 = desc_29 * lower_28 + (1 - desc_29) * (1 - lower_28);
                int tk_31 = ((wmax_30 != 0) ? po_27 : 1 - po_27);
                fk[0] = ((tk_31 != 0) ? pb_24 : fk[0]);
                fi[0] = ((tk_31 != 0) ? pc_25 : fi[0]);
                unsigned int _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 1);
                unsigned int pb_32 = _shfl_xor_106;
                int _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 1);
                int pc_33 = _shfl_xor_107;
                int r_34 = (int)(pb_32 > fk[1] || pb_32 == fk[1] && pc_33 < fi[1]);
                int po_35 = r_34;
                int lower_36 = (int)((lane_0 & 1) == 0);
                int desc_37 = (int)((lane_0 & 4) == 0);
                int wmax_38 = desc_37 * lower_36 + (1 - desc_37) * (1 - lower_36);
                int tk_39 = ((wmax_38 != 0) ? po_35 : 1 - po_35);
                fk[1] = ((tk_39 != 0) ? pb_32 : fk[1]);
                fi[1] = ((tk_39 != 0) ? pc_33 : fi[1]);
                unsigned int _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 4);
                unsigned int pb_40 = _shfl_xor_108;
                int _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 4);
                int pc_41 = _shfl_xor_109;
                int r_42 = (int)(pb_40 > fk[0] || pb_40 == fk[0] && pc_41 < fi[0]);
                int po_43 = r_42;
                int lower_44 = (int)((lane_0 & 4) == 0);
                int desc_45 = (int)((lane_0 & 8) == 0);
                int wmax_46 = desc_45 * lower_44 + (1 - desc_45) * (1 - lower_44);
                int tk_47 = ((wmax_46 != 0) ? po_43 : 1 - po_43);
                fk[0] = ((tk_47 != 0) ? pb_40 : fk[0]);
                fi[0] = ((tk_47 != 0) ? pc_41 : fi[0]);
                unsigned int _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 4);
                unsigned int pb_48 = _shfl_xor_110;
                int _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 4);
                int pc_49 = _shfl_xor_111;
                int r_50 = (int)(pb_48 > fk[1] || pb_48 == fk[1] && pc_49 < fi[1]);
                int po_51 = r_50;
                int lower_52 = (int)((lane_0 & 4) == 0);
                int desc_53 = (int)((lane_0 & 8) == 0);
                int wmax_54 = desc_53 * lower_52 + (1 - desc_53) * (1 - lower_52);
                int tk_55 = ((wmax_54 != 0) ? po_51 : 1 - po_51);
                fk[1] = ((tk_55 != 0) ? pb_48 : fk[1]);
                fi[1] = ((tk_55 != 0) ? pc_49 : fi[1]);
                unsigned int _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 2);
                unsigned int pb_56 = _shfl_xor_112;
                int _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 2);
                int pc_57 = _shfl_xor_113;
                int r_58 = (int)(pb_56 > fk[0] || pb_56 == fk[0] && pc_57 < fi[0]);
                int po_59 = r_58;
                int lower_60 = (int)((lane_0 & 2) == 0);
                int desc_61 = (int)((lane_0 & 8) == 0);
                int wmax_62 = desc_61 * lower_60 + (1 - desc_61) * (1 - lower_60);
                int tk_63 = ((wmax_62 != 0) ? po_59 : 1 - po_59);
                fk[0] = ((tk_63 != 0) ? pb_56 : fk[0]);
                fi[0] = ((tk_63 != 0) ? pc_57 : fi[0]);
                unsigned int _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 2);
                unsigned int pb_64 = _shfl_xor_114;
                int _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 2);
                int pc_65 = _shfl_xor_115;
                int r_66 = (int)(pb_64 > fk[1] || pb_64 == fk[1] && pc_65 < fi[1]);
                int po_67 = r_66;
                int lower_68 = (int)((lane_0 & 2) == 0);
                int desc_69 = (int)((lane_0 & 8) == 0);
                int wmax_70 = desc_69 * lower_68 + (1 - desc_69) * (1 - lower_68);
                int tk_71 = ((wmax_70 != 0) ? po_67 : 1 - po_67);
                fk[1] = ((tk_71 != 0) ? pb_64 : fk[1]);
                fi[1] = ((tk_71 != 0) ? pc_65 : fi[1]);
                unsigned int _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 1);
                unsigned int pb_72 = _shfl_xor_116;
                int _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 1);
                int pc_73 = _shfl_xor_117;
                int r_74 = (int)(pb_72 > fk[0] || pb_72 == fk[0] && pc_73 < fi[0]);
                int po_75 = r_74;
                int lower_76 = (int)((lane_0 & 1) == 0);
                int desc_77 = (int)((lane_0 & 8) == 0);
                int wmax_78 = desc_77 * lower_76 + (1 - desc_77) * (1 - lower_76);
                int tk_79 = ((wmax_78 != 0) ? po_75 : 1 - po_75);
                fk[0] = ((tk_79 != 0) ? pb_72 : fk[0]);
                fi[0] = ((tk_79 != 0) ? pc_73 : fi[0]);
                unsigned int _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 1);
                unsigned int pb_80 = _shfl_xor_118;
                int _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 1);
                int pc_81 = _shfl_xor_119;
                int r_82 = (int)(pb_80 > fk[1] || pb_80 == fk[1] && pc_81 < fi[1]);
                int po_83 = r_82;
                int lower_84 = (int)((lane_0 & 1) == 0);
                int desc_85 = (int)((lane_0 & 8) == 0);
                int wmax_86 = desc_85 * lower_84 + (1 - desc_85) * (1 - lower_84);
                int tk_87 = ((wmax_86 != 0) ? po_83 : 1 - po_83);
                fk[1] = ((tk_87 != 0) ? pb_80 : fk[1]);
                fi[1] = ((tk_87 != 0) ? pc_81 : fi[1]);
                unsigned int _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 8);
                unsigned int pb_88 = _shfl_xor_120;
                int _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 8);
                int pc_89 = _shfl_xor_121;
                int r_90 = (int)(pb_88 > fk[0] || pb_88 == fk[0] && pc_89 < fi[0]);
                int po_91 = r_90;
                int lower_92 = (int)((lane_0 & 8) == 0);
                int desc_93 = (int)((lane_0 & 16) == 0);
                int wmax_94 = desc_93 * lower_92 + (1 - desc_93) * (1 - lower_92);
                int tk_95 = ((wmax_94 != 0) ? po_91 : 1 - po_91);
                fk[0] = ((tk_95 != 0) ? pb_88 : fk[0]);
                fi[0] = ((tk_95 != 0) ? pc_89 : fi[0]);
                unsigned int _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 8);
                unsigned int pb_96 = _shfl_xor_122;
                int _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 8);
                int pc_97 = _shfl_xor_123;
                int r_98 = (int)(pb_96 > fk[1] || pb_96 == fk[1] && pc_97 < fi[1]);
                int po_99 = r_98;
                int lower_100 = (int)((lane_0 & 8) == 0);
                int desc_101 = (int)((lane_0 & 16) == 0);
                int wmax_102 = desc_101 * lower_100 + (1 - desc_101) * (1 - lower_100);
                int tk_103 = ((wmax_102 != 0) ? po_99 : 1 - po_99);
                fk[1] = ((tk_103 != 0) ? pb_96 : fk[1]);
                fi[1] = ((tk_103 != 0) ? pc_97 : fi[1]);
                unsigned int _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 4);
                unsigned int pb_104 = _shfl_xor_124;
                int _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 4);
                int pc_105 = _shfl_xor_125;
                int r_106 = (int)(pb_104 > fk[0] || pb_104 == fk[0] && pc_105 < fi[0]);
                int po_107 = r_106;
                int lower_108 = (int)((lane_0 & 4) == 0);
                int desc_109 = (int)((lane_0 & 16) == 0);
                int wmax_110 = desc_109 * lower_108 + (1 - desc_109) * (1 - lower_108);
                int tk_111 = ((wmax_110 != 0) ? po_107 : 1 - po_107);
                fk[0] = ((tk_111 != 0) ? pb_104 : fk[0]);
                fi[0] = ((tk_111 != 0) ? pc_105 : fi[0]);
                unsigned int _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 4);
                unsigned int pb_112 = _shfl_xor_126;
                int _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 4);
                int pc_113 = _shfl_xor_127;
                int r_114 = (int)(pb_112 > fk[1] || pb_112 == fk[1] && pc_113 < fi[1]);
                int po_115 = r_114;
                int lower_116 = (int)((lane_0 & 4) == 0);
                int desc_117 = (int)((lane_0 & 16) == 0);
                int wmax_118 = desc_117 * lower_116 + (1 - desc_117) * (1 - lower_116);
                int tk_119 = ((wmax_118 != 0) ? po_115 : 1 - po_115);
                fk[1] = ((tk_119 != 0) ? pb_112 : fk[1]);
                fi[1] = ((tk_119 != 0) ? pc_113 : fi[1]);
                unsigned int _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 2);
                unsigned int pb_120_1 = _shfl_xor_128;
                int _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 2);
                int pc_121_1 = _shfl_xor_129;
                int r_122_1 = (int)(pb_120_1 > fk[0] || pb_120_1 == fk[0] && pc_121_1 < fi[0]);
                int po_123_1 = r_122_1;
                int lower_124_1 = (int)((lane_0 & 2) == 0);
                int desc_125_1 = (int)((lane_0 & 16) == 0);
                int wmax_126_1 = desc_125_1 * lower_124_1 + (1 - desc_125_1) * (1 - lower_124_1);
                int tk_127_1 = ((wmax_126_1 != 0) ? po_123_1 : 1 - po_123_1);
                fk[0] = ((tk_127_1 != 0) ? pb_120_1 : fk[0]);
                fi[0] = ((tk_127_1 != 0) ? pc_121_1 : fi[0]);
                unsigned int _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 2);
                unsigned int pb_128_1 = _shfl_xor_130;
                int _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 2);
                int pc_129_1 = _shfl_xor_131;
                int r_130_1 = (int)(pb_128_1 > fk[1] || pb_128_1 == fk[1] && pc_129_1 < fi[1]);
                int po_131_1 = r_130_1;
                int lower_132_1 = (int)((lane_0 & 2) == 0);
                int desc_133_1 = (int)((lane_0 & 16) == 0);
                int wmax_134_1 = desc_133_1 * lower_132_1 + (1 - desc_133_1) * (1 - lower_132_1);
                int tk_135_1 = ((wmax_134_1 != 0) ? po_131_1 : 1 - po_131_1);
                fk[1] = ((tk_135_1 != 0) ? pb_128_1 : fk[1]);
                fi[1] = ((tk_135_1 != 0) ? pc_129_1 : fi[1]);
                unsigned int _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 1);
                unsigned int pb_136_1 = _shfl_xor_132;
                int _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 1);
                int pc_137_1 = _shfl_xor_133;
                int r_138_1 = (int)(pb_136_1 > fk[0] || pb_136_1 == fk[0] && pc_137_1 < fi[0]);
                int po_139_1 = r_138_1;
                int lower_140_1 = (int)((lane_0 & 1) == 0);
                int desc_141_1 = (int)((lane_0 & 16) == 0);
                int wmax_142_1 = desc_141_1 * lower_140_1 + (1 - desc_141_1) * (1 - lower_140_1);
                int tk_143_1 = ((wmax_142_1 != 0) ? po_139_1 : 1 - po_139_1);
                fk[0] = ((tk_143_1 != 0) ? pb_136_1 : fk[0]);
                fi[0] = ((tk_143_1 != 0) ? pc_137_1 : fi[0]);
                unsigned int _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 1);
                unsigned int pb_144_1 = _shfl_xor_134;
                int _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 1);
                int pc_145_1 = _shfl_xor_135;
                int r_146_1 = (int)(pb_144_1 > fk[1] || pb_144_1 == fk[1] && pc_145_1 < fi[1]);
                int po_147_1 = r_146_1;
                int lower_148_1 = (int)((lane_0 & 1) == 0);
                int desc_149_1 = (int)((lane_0 & 16) == 0);
                int wmax_150_1 = desc_149_1 * lower_148_1 + (1 - desc_149_1) * (1 - lower_148_1);
                int tk_151_1 = ((wmax_150_1 != 0) ? po_147_1 : 1 - po_147_1);
                fk[1] = ((tk_151_1 != 0) ? pb_144_1 : fk[1]);
                fi[1] = ((tk_151_1 != 0) ? pc_145_1 : fi[1]);
                unsigned int _shfl_xor_136 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 16);
                unsigned int pb_152_1 = _shfl_xor_136;
                int _shfl_xor_137 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 16);
                int pc_153_1 = _shfl_xor_137;
                int r_154_1 = (int)(pb_152_1 > fk[0] || pb_152_1 == fk[0] && pc_153_1 < fi[0]);
                int po_155_1 = r_154_1;
                int lower_156_1 = (int)((lane_0 & 16) == 0);
                int desc_157_1 = (int)((lane_0 & 0) == 0);
                int wmax_158_1 = desc_157_1 * lower_156_1 + (1 - desc_157_1) * (1 - lower_156_1);
                int tk_159_1 = ((wmax_158_1 != 0) ? po_155_1 : 1 - po_155_1);
                fk[0] = ((tk_159_1 != 0) ? pb_152_1 : fk[0]);
                fi[0] = ((tk_159_1 != 0) ? pc_153_1 : fi[0]);
                unsigned int _shfl_xor_138 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 16);
                unsigned int pb_160_1 = _shfl_xor_138;
                int _shfl_xor_139 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 16);
                int pc_161_1 = _shfl_xor_139;
                int r_162_1 = (int)(pb_160_1 > fk[1] || pb_160_1 == fk[1] && pc_161_1 < fi[1]);
                int po_163_1 = r_162_1;
                int lower_164_1 = (int)((lane_0 & 16) == 0);
                int desc_165_1 = 0;
                int wmax_166_1 = desc_165_1 * lower_164_1 + (1 - desc_165_1) * (1 - lower_164_1);
                int tk_167_1 = ((wmax_166_1 != 0) ? po_163_1 : 1 - po_163_1);
                fk[1] = ((tk_167_1 != 0) ? pb_160_1 : fk[1]);
                fi[1] = ((tk_167_1 != 0) ? pc_161_1 : fi[1]);
                unsigned int _shfl_xor_140 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 8);
                unsigned int pb_168_1 = _shfl_xor_140;
                int _shfl_xor_141 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 8);
                int pc_169_1 = _shfl_xor_141;
                int r_170_1 = (int)(pb_168_1 > fk[0] || pb_168_1 == fk[0] && pc_169_1 < fi[0]);
                int po_171_1 = r_170_1;
                int lower_172_1 = (int)((lane_0 & 8) == 0);
                int desc_173_1 = (int)((lane_0 & 0) == 0);
                int wmax_174_1 = desc_173_1 * lower_172_1 + (1 - desc_173_1) * (1 - lower_172_1);
                int tk_175_1 = ((wmax_174_1 != 0) ? po_171_1 : 1 - po_171_1);
                fk[0] = ((tk_175_1 != 0) ? pb_168_1 : fk[0]);
                fi[0] = ((tk_175_1 != 0) ? pc_169_1 : fi[0]);
                unsigned int _shfl_xor_142 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 8);
                unsigned int pb_176_1 = _shfl_xor_142;
                int _shfl_xor_143 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 8);
                int pc_177_1 = _shfl_xor_143;
                int r_178_1 = (int)(pb_176_1 > fk[1] || pb_176_1 == fk[1] && pc_177_1 < fi[1]);
                int po_179_1 = r_178_1;
                int lower_180_1 = (int)((lane_0 & 8) == 0);
                int desc_181_1 = 0;
                int wmax_182_1 = desc_181_1 * lower_180_1 + (1 - desc_181_1) * (1 - lower_180_1);
                int tk_183_1 = ((wmax_182_1 != 0) ? po_179_1 : 1 - po_179_1);
                fk[1] = ((tk_183_1 != 0) ? pb_176_1 : fk[1]);
                fi[1] = ((tk_183_1 != 0) ? pc_177_1 : fi[1]);
                unsigned int _shfl_xor_144 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 4);
                unsigned int pb_184_1 = _shfl_xor_144;
                int _shfl_xor_145 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 4);
                int pc_185_1 = _shfl_xor_145;
                int r_186_1 = (int)(pb_184_1 > fk[0] || pb_184_1 == fk[0] && pc_185_1 < fi[0]);
                int po_187_1 = r_186_1;
                int lower_188_1 = (int)((lane_0 & 4) == 0);
                int desc_189_1 = (int)((lane_0 & 0) == 0);
                int wmax_190_1 = desc_189_1 * lower_188_1 + (1 - desc_189_1) * (1 - lower_188_1);
                int tk_191_1 = ((wmax_190_1 != 0) ? po_187_1 : 1 - po_187_1);
                fk[0] = ((tk_191_1 != 0) ? pb_184_1 : fk[0]);
                fi[0] = ((tk_191_1 != 0) ? pc_185_1 : fi[0]);
                unsigned int _shfl_xor_146 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 4);
                unsigned int pb_192_1 = _shfl_xor_146;
                int _shfl_xor_147 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 4);
                int pc_193_1 = _shfl_xor_147;
                int r_194_1 = (int)(pb_192_1 > fk[1] || pb_192_1 == fk[1] && pc_193_1 < fi[1]);
                int po_195_1 = r_194_1;
                int lower_196_1 = (int)((lane_0 & 4) == 0);
                int desc_197_1 = 0;
                int wmax_198_1 = desc_197_1 * lower_196_1 + (1 - desc_197_1) * (1 - lower_196_1);
                int tk_199_1 = ((wmax_198_1 != 0) ? po_195_1 : 1 - po_195_1);
                fk[1] = ((tk_199_1 != 0) ? pb_192_1 : fk[1]);
                fi[1] = ((tk_199_1 != 0) ? pc_193_1 : fi[1]);
                unsigned int _shfl_xor_148 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 2);
                unsigned int pb_200_1 = _shfl_xor_148;
                int _shfl_xor_149 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 2);
                int pc_201_1 = _shfl_xor_149;
                int r_202_1 = (int)(pb_200_1 > fk[0] || pb_200_1 == fk[0] && pc_201_1 < fi[0]);
                int po_203_1 = r_202_1;
                int lower_204_1 = (int)((lane_0 & 2) == 0);
                int desc_205_1 = (int)((lane_0 & 0) == 0);
                int wmax_206_1 = desc_205_1 * lower_204_1 + (1 - desc_205_1) * (1 - lower_204_1);
                int tk_207_1 = ((wmax_206_1 != 0) ? po_203_1 : 1 - po_203_1);
                fk[0] = ((tk_207_1 != 0) ? pb_200_1 : fk[0]);
                fi[0] = ((tk_207_1 != 0) ? pc_201_1 : fi[0]);
                unsigned int _shfl_xor_150 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 2);
                unsigned int pb_208_1 = _shfl_xor_150;
                int _shfl_xor_151 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 2);
                int pc_209_1 = _shfl_xor_151;
                int r_210_1 = (int)(pb_208_1 > fk[1] || pb_208_1 == fk[1] && pc_209_1 < fi[1]);
                int po_211_1 = r_210_1;
                int lower_212_1 = (int)((lane_0 & 2) == 0);
                int desc_213_1 = 0;
                int wmax_214_1 = desc_213_1 * lower_212_1 + (1 - desc_213_1) * (1 - lower_212_1);
                int tk_215_1 = ((wmax_214_1 != 0) ? po_211_1 : 1 - po_211_1);
                fk[1] = ((tk_215_1 != 0) ? pb_208_1 : fk[1]);
                fi[1] = ((tk_215_1 != 0) ? pc_209_1 : fi[1]);
                unsigned int _shfl_xor_152 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 1);
                unsigned int pb_216_1 = _shfl_xor_152;
                int _shfl_xor_153 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 1);
                int pc_217_1 = _shfl_xor_153;
                int r_218_1 = (int)(pb_216_1 > fk[0] || pb_216_1 == fk[0] && pc_217_1 < fi[0]);
                int po_219_1 = r_218_1;
                int lower_220_1 = (int)((lane_0 & 1) == 0);
                int desc_221_1 = (int)((lane_0 & 0) == 0);
                int wmax_222_1 = desc_221_1 * lower_220_1 + (1 - desc_221_1) * (1 - lower_220_1);
                int tk_223_1 = ((wmax_222_1 != 0) ? po_219_1 : 1 - po_219_1);
                fk[0] = ((tk_223_1 != 0) ? pb_216_1 : fk[0]);
                fi[0] = ((tk_223_1 != 0) ? pc_217_1 : fi[0]);
                unsigned int _shfl_xor_154 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 1);
                unsigned int pb_224_1 = _shfl_xor_154;
                int _shfl_xor_155 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 1);
                int pc_225_1 = _shfl_xor_155;
                int r_226_1 = (int)(pb_224_1 > fk[1] || pb_224_1 == fk[1] && pc_225_1 < fi[1]);
                int po_227_1 = r_226_1;
                int lower_228_1 = (int)((lane_0 & 1) == 0);
                int desc_229_1 = 0;
                int wmax_230_1 = desc_229_1 * lower_228_1 + (1 - desc_229_1) * (1 - lower_228_1);
                int tk_231_1 = ((wmax_230_1 != 0) ? po_227_1 : 1 - po_227_1);
                fk[1] = ((tk_231_1 != 0) ? pb_224_1 : fk[1]);
                fi[1] = ((tk_231_1 != 0) ? pc_225_1 : fi[1]);
                unsigned int xk = fk[0];
                int xi = fi[0];
                unsigned int yk_1 = fk[1];
                int yi_1 = fi[1];
                int take = (int)(yk_1 > xk || yk_1 == xk && yi_1 < xi);
                fk[0] = ((take != 0) ? yk_1 : xk);
                fi[0] = ((take != 0) ? yi_1 : xi);
                fk[1] = ((take != 0) ? xk : yk_1);
                fi[1] = ((take != 0) ? xi : yi_1);
                unsigned int _shfl_xor_156 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 16);
                unsigned int pb_232_1 = _shfl_xor_156;
                int _shfl_xor_157 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 16);
                int pc_233_1 = _shfl_xor_157;
                int r_234_1 = (int)(pb_232_1 > fk[0] || pb_232_1 == fk[0] && pc_233_1 < fi[0]);
                int po_235_1 = r_234_1;
                int lower_236_1 = (int)((lane_0 & 16) == 0);
                int desc_237_1 = (int)((lane_0 & 0) == 0);
                int wmax_238_1 = desc_237_1 * lower_236_1 + (1 - desc_237_1) * (1 - lower_236_1);
                int tk_239_1 = ((wmax_238_1 != 0) ? po_235_1 : 1 - po_235_1);
                fk[0] = ((tk_239_1 != 0) ? pb_232_1 : fk[0]);
                fi[0] = ((tk_239_1 != 0) ? pc_233_1 : fi[0]);
                unsigned int _shfl_xor_158 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 16);
                unsigned int pb_240 = _shfl_xor_158;
                int _shfl_xor_159 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 16);
                int pc_241 = _shfl_xor_159;
                int r_242 = (int)(pb_240 > fk[1] || pb_240 == fk[1] && pc_241 < fi[1]);
                int po_243 = r_242;
                int lower_244 = (int)((lane_0 & 16) == 0);
                int desc_245 = (int)((lane_0 & 0) == 0);
                int wmax_246 = desc_245 * lower_244 + (1 - desc_245) * (1 - lower_244);
                int tk_247 = ((wmax_246 != 0) ? po_243 : 1 - po_243);
                fk[1] = ((tk_247 != 0) ? pb_240 : fk[1]);
                fi[1] = ((tk_247 != 0) ? pc_241 : fi[1]);
                unsigned int _shfl_xor_160 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 8);
                unsigned int pb_248 = _shfl_xor_160;
                int _shfl_xor_161 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 8);
                int pc_249 = _shfl_xor_161;
                int r_250 = (int)(pb_248 > fk[0] || pb_248 == fk[0] && pc_249 < fi[0]);
                int po_251 = r_250;
                int lower_252 = (int)((lane_0 & 8) == 0);
                int desc_253 = (int)((lane_0 & 0) == 0);
                int wmax_254 = desc_253 * lower_252 + (1 - desc_253) * (1 - lower_252);
                int tk_255 = ((wmax_254 != 0) ? po_251 : 1 - po_251);
                fk[0] = ((tk_255 != 0) ? pb_248 : fk[0]);
                fi[0] = ((tk_255 != 0) ? pc_249 : fi[0]);
                unsigned int _shfl_xor_162 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 8);
                unsigned int pb_256 = _shfl_xor_162;
                int _shfl_xor_163 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 8);
                int pc_257 = _shfl_xor_163;
                int r_258 = (int)(pb_256 > fk[1] || pb_256 == fk[1] && pc_257 < fi[1]);
                int po_259 = r_258;
                int lower_260 = (int)((lane_0 & 8) == 0);
                int desc_261 = (int)((lane_0 & 0) == 0);
                int wmax_262 = desc_261 * lower_260 + (1 - desc_261) * (1 - lower_260);
                int tk_263 = ((wmax_262 != 0) ? po_259 : 1 - po_259);
                fk[1] = ((tk_263 != 0) ? pb_256 : fk[1]);
                fi[1] = ((tk_263 != 0) ? pc_257 : fi[1]);
                unsigned int _shfl_xor_164 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 4);
                unsigned int pb_264 = _shfl_xor_164;
                int _shfl_xor_165 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 4);
                int pc_265 = _shfl_xor_165;
                int r_266 = (int)(pb_264 > fk[0] || pb_264 == fk[0] && pc_265 < fi[0]);
                int po_267 = r_266;
                int lower_268 = (int)((lane_0 & 4) == 0);
                int desc_269 = (int)((lane_0 & 0) == 0);
                int wmax_270 = desc_269 * lower_268 + (1 - desc_269) * (1 - lower_268);
                int tk_271 = ((wmax_270 != 0) ? po_267 : 1 - po_267);
                fk[0] = ((tk_271 != 0) ? pb_264 : fk[0]);
                fi[0] = ((tk_271 != 0) ? pc_265 : fi[0]);
                unsigned int _shfl_xor_166 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 4);
                unsigned int pb_272 = _shfl_xor_166;
                int _shfl_xor_167 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 4);
                int pc_273 = _shfl_xor_167;
                int r_274 = (int)(pb_272 > fk[1] || pb_272 == fk[1] && pc_273 < fi[1]);
                int po_275 = r_274;
                int lower_276 = (int)((lane_0 & 4) == 0);
                int desc_277 = (int)((lane_0 & 0) == 0);
                int wmax_278 = desc_277 * lower_276 + (1 - desc_277) * (1 - lower_276);
                int tk_279 = ((wmax_278 != 0) ? po_275 : 1 - po_275);
                fk[1] = ((tk_279 != 0) ? pb_272 : fk[1]);
                fi[1] = ((tk_279 != 0) ? pc_273 : fi[1]);
                unsigned int _shfl_xor_168 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 2);
                unsigned int pb_280 = _shfl_xor_168;
                int _shfl_xor_169 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 2);
                int pc_281 = _shfl_xor_169;
                int r_282 = (int)(pb_280 > fk[0] || pb_280 == fk[0] && pc_281 < fi[0]);
                int po_283 = r_282;
                int lower_284 = (int)((lane_0 & 2) == 0);
                int desc_285 = (int)((lane_0 & 0) == 0);
                int wmax_286 = desc_285 * lower_284 + (1 - desc_285) * (1 - lower_284);
                int tk_287 = ((wmax_286 != 0) ? po_283 : 1 - po_283);
                fk[0] = ((tk_287 != 0) ? pb_280 : fk[0]);
                fi[0] = ((tk_287 != 0) ? pc_281 : fi[0]);
                unsigned int _shfl_xor_170 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 2);
                unsigned int pb_288 = _shfl_xor_170;
                int _shfl_xor_171 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 2);
                int pc_289 = _shfl_xor_171;
                int r_290 = (int)(pb_288 > fk[1] || pb_288 == fk[1] && pc_289 < fi[1]);
                int po_291 = r_290;
                int lower_292 = (int)((lane_0 & 2) == 0);
                int desc_293 = (int)((lane_0 & 0) == 0);
                int wmax_294 = desc_293 * lower_292 + (1 - desc_293) * (1 - lower_292);
                int tk_295 = ((wmax_294 != 0) ? po_291 : 1 - po_291);
                fk[1] = ((tk_295 != 0) ? pb_288 : fk[1]);
                fi[1] = ((tk_295 != 0) ? pc_289 : fi[1]);
                unsigned int _shfl_xor_172 = __shfl_xor_sync(0xFFFFFFFF, fk[0], 1);
                unsigned int pb_296 = _shfl_xor_172;
                int _shfl_xor_173 = __shfl_xor_sync(0xFFFFFFFF, fi[0], 1);
                int pc_297 = _shfl_xor_173;
                int r_298 = (int)(pb_296 > fk[0] || pb_296 == fk[0] && pc_297 < fi[0]);
                int po_299 = r_298;
                int lower_300 = (int)((lane_0 & 1) == 0);
                int desc_301 = (int)((lane_0 & 0) == 0);
                int wmax_302 = desc_301 * lower_300 + (1 - desc_301) * (1 - lower_300);
                int tk_303 = ((wmax_302 != 0) ? po_299 : 1 - po_299);
                fk[0] = ((tk_303 != 0) ? pb_296 : fk[0]);
                fi[0] = ((tk_303 != 0) ? pc_297 : fi[0]);
                unsigned int _shfl_xor_174 = __shfl_xor_sync(0xFFFFFFFF, fk[1], 1);
                unsigned int pb_304 = _shfl_xor_174;
                int _shfl_xor_175 = __shfl_xor_sync(0xFFFFFFFF, fi[1], 1);
                int pc_305 = _shfl_xor_175;
                int r_306 = (int)(pb_304 > fk[1] || pb_304 == fk[1] && pc_305 < fi[1]);
                int po_307 = r_306;
                int lower_308 = (int)((lane_0 & 1) == 0);
                int desc_309 = (int)((lane_0 & 0) == 0);
                int wmax_310 = desc_309 * lower_308 + (1 - desc_309) * (1 - lower_308);
                int tk_311 = ((wmax_310 != 0) ? po_307 : 1 - po_307);
                fk[1] = ((tk_311 != 0) ? pb_304 : fk[1]);
                fi[1] = ((tk_311 != 0) ? pc_305 : fi[1]);
                hk[0] = fk[0];
                hi[0] = fi[0];
                rnk = lane_0;
                h_real = (int)(hk[0] != 8388607 || hi[0] != 2147483647);
                unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, h_real != 0);
                int _popc_0 = __popc(_vote_0);
                S = _popc_0;
            }
            more = 0;
        }
    }
    if (h_real != 0 && rnk < 16) {
        carry_k[rnk] = hk[0];
        carry_i[rnk] = hi[0];
    }
    if (tr >= S && tr < 16) {
        carry_k[tr] = 8388607;
        carry_i[tr] = 2147483647;
    }
    __syncwarp();
    long long c_out = (long long)row * 32;
    if (h_real != 0 && rnk < 16) {
        unsigned int bits = ~hk[0];
        if ((hk[0] & 2147483648u) != 0) {
            bits = hk[0] ^ 2147483648u;
        }
        int s_bits = 0;
        s_bits = reinterpret_cast<int*>(&bits)[0];
        int out2[2];
        out2[0] = s_bits;
        out2[1] = hi[0];
        {
            int2 _iv2 = make_int2(out2[0 + 0], out2[0 + 1]);
            *reinterpret_cast<int2*>(candidates + (c_out + (long long)rnk * 2) + 0) = _iv2;
        }
    }
    if (tr >= S && tr < 16) {
        int pad2[2];
        pad2[0] = -8388608;
        pad2[1] = -1;
        {
            int2 _iv2 = make_int2(pad2[0 + 0], pad2[0 + 1]);
            *reinterpret_cast<int2*>(candidates + (c_out + (long long)tr * 2) + 0) = _iv2;
        }
    }
    int t_tok = row / 4;
    int h_x = row - t_tok * 4;
    int owner = h_x / 2;
    int row_x = t_tok * 2 + (h_x - owner * 2);
    long long x_row = (long long)(g & 1) * (long long)x_par_stride + (long long)row_x * 32;
    unsigned long long t_ready = t0;
    if (sub == 0) {
        if (x_phase != 2 && owner != rank) {
            if (lane_0 < 16) {
                unsigned int pk = carry_k[lane_0];
                int pi = carry_i[lane_0];
                unsigned int pid = (unsigned int)pi & 1048575;
                if (pi == 2147483647) {
                    pid = 1048575;
                }
                unsigned int p_hi = (unsigned int)g << 20 | pid;
                unsigned long long p_word = (unsigned long long)p_hi << 32 | (unsigned long long)pk;
                asm volatile("st.volatile.global.u64 [%0], %1;" :: "l"(&xpool_peers.ptrs[owner][x_row + (long long)(rank * 16 + lane_0)]), "l"(static_cast<uint64_t>(p_word)) : "memory");
            }
        }
        unsigned long long gtimer_0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_0) :: "memory");
        t_ready = gtimer_0;
        if (owner == rank && x_phase != 1) {
            int fb = forced_block[token_begin + t_tok];
            int n_ord = n_ordinary[token_begin + t_tok];
            int act = (int)(lane_0 < 8);
            int wi = 2 * lane_0;
            int pj = wi / 16;
            int src = ((pj < rank) ? pj : pj + 1);
            int wo = wi - pj * 16;
            long long ad = ((act != 0) ? x_row + (long long)(src * 16 + wo) : x_row);
            unsigned long long wv_1[2];
            {
                unsigned long long _u64x2_1_0;
                unsigned long long _u64x2_1_1;
                asm volatile("ld.volatile.global.v2.u64 {%0, %1}, [%2];"
                    : "=l"(_u64x2_1_0), "=l"(_u64x2_1_1) : "l"((const void*)(xpool + (ad))) : "memory");
                wv_1[0 + 0] = _u64x2_1_0;
                wv_1[0 + 1] = _u64x2_1_1;
            }
            int g0 = (int)((unsigned int)(wv_1[0] >> 32) >> 20);
            int g1 = (int)((unsigned int)(wv_1[1] >> 32) >> 20);
            int bad = act & (int)(g0 != g || g1 != g);
            int _vote_1 = __any_sync(0xFFFFFFFF, bad != 0);
            int anyb = _vote_1;
            int timed = 0;
            while (anyb != 0) {
                {
                    unsigned long long _u64x2_2_0;
                    unsigned long long _u64x2_2_1;
                    asm volatile("ld.volatile.global.v2.u64 {%0, %1}, [%2];"
                        : "=l"(_u64x2_2_0), "=l"(_u64x2_2_1) : "l"((const void*)(xpool + (ad))) : "memory");
                    wv_1[0 + 0] = _u64x2_2_0;
                    wv_1[0 + 1] = _u64x2_2_1;
                }
                g0 = (int)((unsigned int)(wv_1[0] >> 32) >> 20);
                g1 = (int)((unsigned int)(wv_1[1] >> 32) >> 20);
                bad = act & (int)(g0 != g || g1 != g);
                unsigned long long gtimer_2;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_2) :: "memory");
                unsigned long long now = gtimer_2;
                long long el = (long long)(now - t0);
                if (bad != 0 && el > timeout_ns) {
                    timed = 1;
                    bad = 0;
                }
                int _vote_2 = __any_sync(0xFFFFFFFF, bad != 0);
                anyb = _vote_2;
            }
            if (timed != 0) {
                status[0] = 8;
            }
            unsigned long long gtimer_2_1;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_2_1) :: "memory");
            t_ready = gtimer_2_1;
            int s_me = lane_0 / 16;
            int q_me = lane_0 - s_me * 16;
            unsigned long long pzw = -1;
            unsigned long long _shfl_2 = __shfl_sync(0xFFFFFFFF, wv_1[0], q_me / 2);
            unsigned long long wq0 = _shfl_2;
            unsigned long long _shfl_3 = __shfl_sync(0xFFFFFFFF, wv_1[1], q_me / 2);
            unsigned long long wq1 = _shfl_3;
            unsigned long long wq = (((q_me & 1) != 0) ? wq1 : wq0);
            unsigned int key_r = (unsigned int)wq;
            unsigned int whi_r = (unsigned int)(wq >> 32);
            int idf_r = (int)(whi_r & 1048575);
            int gq_r = (int)(whi_r >> 20);
            int live_r = (int)(gq_r == g && idf_r != 1048575 && idf_r != fb);
            int packed = idf_r << 1 | live_r;
            if (idf_r == 1048575) {
                packed = 2147483646;
            }
            if (s_me == rank) {
                unsigned int ck_r = carry_k[q_me];
                int ci_r = carry_i[q_me];
                key_r = ck_r;
                live_r = (int)(ci_r != 2147483647 && ci_r != fb);
                packed = (ci_r & 1048575) << 1 | live_r;
                if (ci_r == 2147483647) {
                    packed = 2147483646;
                }
            }
            if (act != 0) {
                asm volatile("st.volatile.global.u64 [%0], %1;" :: "l"(&xpool[ad]), "l"(static_cast<uint64_t>(pzw)) : "memory");
                asm volatile("st.volatile.global.u64 [%0], %1;" :: "l"(&xpool[ad + 1]), "l"(static_cast<uint64_t>(pzw)) : "memory");
            }
            int lose_r = 0;
            int _shfl_4 = __shfl_sync(0xFFFFFFFF, packed, 0);
            int pk0 = _shfl_4;
            unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, key_r, 0);
            unsigned int kk0 = _shfl_5;
            int _shfl_6 = __shfl_sync(0xFFFFFFFF, packed, 16);
            int pk1 = _shfl_6;
            unsigned int _shfl_7 = __shfl_sync(0xFFFFFFFF, key_r, 16);
            unsigned int kk1 = _shfl_7;
            lose_r = lose_r | (int)(pk0 == packed) & ((int)(kk0 > key_r) | (int)(kk0 == key_r) & (int)(lane_0 > 0));
            lose_r = lose_r | (int)(pk1 == packed) & ((int)(kk1 > key_r) | (int)(kk1 == key_r) & (int)(lane_0 > 16));
            int _shfl_8 = __shfl_sync(0xFFFFFFFF, packed, 1);
            int pk0_3 = _shfl_8;
            unsigned int _shfl_9 = __shfl_sync(0xFFFFFFFF, key_r, 1);
            unsigned int kk0_4 = _shfl_9;
            int _shfl_10 = __shfl_sync(0xFFFFFFFF, packed, 17);
            int pk1_5 = _shfl_10;
            unsigned int _shfl_11 = __shfl_sync(0xFFFFFFFF, key_r, 17);
            unsigned int kk1_6 = _shfl_11;
            lose_r = lose_r | (int)(pk0_3 == packed) & ((int)(kk0_4 > key_r) | (int)(kk0_4 == key_r) & (int)(lane_0 > 1));
            lose_r = lose_r | (int)(pk1_5 == packed) & ((int)(kk1_6 > key_r) | (int)(kk1_6 == key_r) & (int)(lane_0 > 17));
            int _shfl_12 = __shfl_sync(0xFFFFFFFF, packed, 2);
            int pk0_7 = _shfl_12;
            unsigned int _shfl_13 = __shfl_sync(0xFFFFFFFF, key_r, 2);
            unsigned int kk0_8 = _shfl_13;
            int _shfl_14 = __shfl_sync(0xFFFFFFFF, packed, 18);
            int pk1_9 = _shfl_14;
            unsigned int _shfl_15 = __shfl_sync(0xFFFFFFFF, key_r, 18);
            unsigned int kk1_10 = _shfl_15;
            lose_r = lose_r | (int)(pk0_7 == packed) & ((int)(kk0_8 > key_r) | (int)(kk0_8 == key_r) & (int)(lane_0 > 2));
            lose_r = lose_r | (int)(pk1_9 == packed) & ((int)(kk1_10 > key_r) | (int)(kk1_10 == key_r) & (int)(lane_0 > 18));
            int _shfl_16 = __shfl_sync(0xFFFFFFFF, packed, 3);
            int pk0_11 = _shfl_16;
            unsigned int _shfl_17 = __shfl_sync(0xFFFFFFFF, key_r, 3);
            unsigned int kk0_12 = _shfl_17;
            int _shfl_18 = __shfl_sync(0xFFFFFFFF, packed, 19);
            int pk1_13 = _shfl_18;
            unsigned int _shfl_19 = __shfl_sync(0xFFFFFFFF, key_r, 19);
            unsigned int kk1_14 = _shfl_19;
            lose_r = lose_r | (int)(pk0_11 == packed) & ((int)(kk0_12 > key_r) | (int)(kk0_12 == key_r) & (int)(lane_0 > 3));
            lose_r = lose_r | (int)(pk1_13 == packed) & ((int)(kk1_14 > key_r) | (int)(kk1_14 == key_r) & (int)(lane_0 > 19));
            int _shfl_20 = __shfl_sync(0xFFFFFFFF, packed, 4);
            int pk0_15 = _shfl_20;
            unsigned int _shfl_21 = __shfl_sync(0xFFFFFFFF, key_r, 4);
            unsigned int kk0_16 = _shfl_21;
            int _shfl_22 = __shfl_sync(0xFFFFFFFF, packed, 20);
            int pk1_17 = _shfl_22;
            unsigned int _shfl_23 = __shfl_sync(0xFFFFFFFF, key_r, 20);
            unsigned int kk1_18 = _shfl_23;
            lose_r = lose_r | (int)(pk0_15 == packed) & ((int)(kk0_16 > key_r) | (int)(kk0_16 == key_r) & (int)(lane_0 > 4));
            lose_r = lose_r | (int)(pk1_17 == packed) & ((int)(kk1_18 > key_r) | (int)(kk1_18 == key_r) & (int)(lane_0 > 20));
            int _shfl_24 = __shfl_sync(0xFFFFFFFF, packed, 5);
            int pk0_19 = _shfl_24;
            unsigned int _shfl_25 = __shfl_sync(0xFFFFFFFF, key_r, 5);
            unsigned int kk0_20 = _shfl_25;
            int _shfl_26 = __shfl_sync(0xFFFFFFFF, packed, 21);
            int pk1_21 = _shfl_26;
            unsigned int _shfl_27 = __shfl_sync(0xFFFFFFFF, key_r, 21);
            unsigned int kk1_22 = _shfl_27;
            lose_r = lose_r | (int)(pk0_19 == packed) & ((int)(kk0_20 > key_r) | (int)(kk0_20 == key_r) & (int)(lane_0 > 5));
            lose_r = lose_r | (int)(pk1_21 == packed) & ((int)(kk1_22 > key_r) | (int)(kk1_22 == key_r) & (int)(lane_0 > 21));
            int _shfl_28 = __shfl_sync(0xFFFFFFFF, packed, 6);
            int pk0_23 = _shfl_28;
            unsigned int _shfl_29 = __shfl_sync(0xFFFFFFFF, key_r, 6);
            unsigned int kk0_24 = _shfl_29;
            int _shfl_30 = __shfl_sync(0xFFFFFFFF, packed, 22);
            int pk1_25 = _shfl_30;
            unsigned int _shfl_31 = __shfl_sync(0xFFFFFFFF, key_r, 22);
            unsigned int kk1_26 = _shfl_31;
            lose_r = lose_r | (int)(pk0_23 == packed) & ((int)(kk0_24 > key_r) | (int)(kk0_24 == key_r) & (int)(lane_0 > 6));
            lose_r = lose_r | (int)(pk1_25 == packed) & ((int)(kk1_26 > key_r) | (int)(kk1_26 == key_r) & (int)(lane_0 > 22));
            int _shfl_32 = __shfl_sync(0xFFFFFFFF, packed, 7);
            int pk0_27 = _shfl_32;
            unsigned int _shfl_33 = __shfl_sync(0xFFFFFFFF, key_r, 7);
            unsigned int kk0_28 = _shfl_33;
            int _shfl_34 = __shfl_sync(0xFFFFFFFF, packed, 23);
            int pk1_29 = _shfl_34;
            unsigned int _shfl_35 = __shfl_sync(0xFFFFFFFF, key_r, 23);
            unsigned int kk1_30 = _shfl_35;
            lose_r = lose_r | (int)(pk0_27 == packed) & ((int)(kk0_28 > key_r) | (int)(kk0_28 == key_r) & (int)(lane_0 > 7));
            lose_r = lose_r | (int)(pk1_29 == packed) & ((int)(kk1_30 > key_r) | (int)(kk1_30 == key_r) & (int)(lane_0 > 23));
            int _shfl_36 = __shfl_sync(0xFFFFFFFF, packed, 8);
            int pk0_31 = _shfl_36;
            unsigned int _shfl_37 = __shfl_sync(0xFFFFFFFF, key_r, 8);
            unsigned int kk0_32 = _shfl_37;
            int _shfl_38 = __shfl_sync(0xFFFFFFFF, packed, 24);
            int pk1_33 = _shfl_38;
            unsigned int _shfl_39 = __shfl_sync(0xFFFFFFFF, key_r, 24);
            unsigned int kk1_34 = _shfl_39;
            lose_r = lose_r | (int)(pk0_31 == packed) & ((int)(kk0_32 > key_r) | (int)(kk0_32 == key_r) & (int)(lane_0 > 8));
            lose_r = lose_r | (int)(pk1_33 == packed) & ((int)(kk1_34 > key_r) | (int)(kk1_34 == key_r) & (int)(lane_0 > 24));
            int _shfl_40 = __shfl_sync(0xFFFFFFFF, packed, 9);
            int pk0_35 = _shfl_40;
            unsigned int _shfl_41 = __shfl_sync(0xFFFFFFFF, key_r, 9);
            unsigned int kk0_36 = _shfl_41;
            int _shfl_42 = __shfl_sync(0xFFFFFFFF, packed, 25);
            int pk1_37 = _shfl_42;
            unsigned int _shfl_43 = __shfl_sync(0xFFFFFFFF, key_r, 25);
            unsigned int kk1_38 = _shfl_43;
            lose_r = lose_r | (int)(pk0_35 == packed) & ((int)(kk0_36 > key_r) | (int)(kk0_36 == key_r) & (int)(lane_0 > 9));
            lose_r = lose_r | (int)(pk1_37 == packed) & ((int)(kk1_38 > key_r) | (int)(kk1_38 == key_r) & (int)(lane_0 > 25));
            int _shfl_44 = __shfl_sync(0xFFFFFFFF, packed, 10);
            int pk0_39 = _shfl_44;
            unsigned int _shfl_45 = __shfl_sync(0xFFFFFFFF, key_r, 10);
            unsigned int kk0_40 = _shfl_45;
            int _shfl_46 = __shfl_sync(0xFFFFFFFF, packed, 26);
            int pk1_41 = _shfl_46;
            unsigned int _shfl_47 = __shfl_sync(0xFFFFFFFF, key_r, 26);
            unsigned int kk1_42 = _shfl_47;
            lose_r = lose_r | (int)(pk0_39 == packed) & ((int)(kk0_40 > key_r) | (int)(kk0_40 == key_r) & (int)(lane_0 > 10));
            lose_r = lose_r | (int)(pk1_41 == packed) & ((int)(kk1_42 > key_r) | (int)(kk1_42 == key_r) & (int)(lane_0 > 26));
            int _shfl_48 = __shfl_sync(0xFFFFFFFF, packed, 11);
            int pk0_43 = _shfl_48;
            unsigned int _shfl_49 = __shfl_sync(0xFFFFFFFF, key_r, 11);
            unsigned int kk0_44 = _shfl_49;
            int _shfl_50 = __shfl_sync(0xFFFFFFFF, packed, 27);
            int pk1_45 = _shfl_50;
            unsigned int _shfl_51 = __shfl_sync(0xFFFFFFFF, key_r, 27);
            unsigned int kk1_46 = _shfl_51;
            lose_r = lose_r | (int)(pk0_43 == packed) & ((int)(kk0_44 > key_r) | (int)(kk0_44 == key_r) & (int)(lane_0 > 11));
            lose_r = lose_r | (int)(pk1_45 == packed) & ((int)(kk1_46 > key_r) | (int)(kk1_46 == key_r) & (int)(lane_0 > 27));
            int _shfl_52 = __shfl_sync(0xFFFFFFFF, packed, 12);
            int pk0_47 = _shfl_52;
            unsigned int _shfl_53 = __shfl_sync(0xFFFFFFFF, key_r, 12);
            unsigned int kk0_48 = _shfl_53;
            int _shfl_54 = __shfl_sync(0xFFFFFFFF, packed, 28);
            int pk1_49 = _shfl_54;
            unsigned int _shfl_55 = __shfl_sync(0xFFFFFFFF, key_r, 28);
            unsigned int kk1_50 = _shfl_55;
            lose_r = lose_r | (int)(pk0_47 == packed) & ((int)(kk0_48 > key_r) | (int)(kk0_48 == key_r) & (int)(lane_0 > 12));
            lose_r = lose_r | (int)(pk1_49 == packed) & ((int)(kk1_50 > key_r) | (int)(kk1_50 == key_r) & (int)(lane_0 > 28));
            int _shfl_56 = __shfl_sync(0xFFFFFFFF, packed, 13);
            int pk0_51 = _shfl_56;
            unsigned int _shfl_57 = __shfl_sync(0xFFFFFFFF, key_r, 13);
            unsigned int kk0_52 = _shfl_57;
            int _shfl_58 = __shfl_sync(0xFFFFFFFF, packed, 29);
            int pk1_53 = _shfl_58;
            unsigned int _shfl_59 = __shfl_sync(0xFFFFFFFF, key_r, 29);
            unsigned int kk1_54 = _shfl_59;
            lose_r = lose_r | (int)(pk0_51 == packed) & ((int)(kk0_52 > key_r) | (int)(kk0_52 == key_r) & (int)(lane_0 > 13));
            lose_r = lose_r | (int)(pk1_53 == packed) & ((int)(kk1_54 > key_r) | (int)(kk1_54 == key_r) & (int)(lane_0 > 29));
            int _shfl_60 = __shfl_sync(0xFFFFFFFF, packed, 14);
            int pk0_55 = _shfl_60;
            unsigned int _shfl_61 = __shfl_sync(0xFFFFFFFF, key_r, 14);
            unsigned int kk0_56 = _shfl_61;
            int _shfl_62 = __shfl_sync(0xFFFFFFFF, packed, 30);
            int pk1_57 = _shfl_62;
            unsigned int _shfl_63 = __shfl_sync(0xFFFFFFFF, key_r, 30);
            unsigned int kk1_58 = _shfl_63;
            lose_r = lose_r | (int)(pk0_55 == packed) & ((int)(kk0_56 > key_r) | (int)(kk0_56 == key_r) & (int)(lane_0 > 14));
            lose_r = lose_r | (int)(pk1_57 == packed) & ((int)(kk1_58 > key_r) | (int)(kk1_58 == key_r) & (int)(lane_0 > 30));
            int _shfl_64 = __shfl_sync(0xFFFFFFFF, packed, 15);
            int pk0_59 = _shfl_64;
            unsigned int _shfl_65 = __shfl_sync(0xFFFFFFFF, key_r, 15);
            unsigned int kk0_60 = _shfl_65;
            int _shfl_66 = __shfl_sync(0xFFFFFFFF, packed, 31);
            int pk1_61 = _shfl_66;
            unsigned int _shfl_67 = __shfl_sync(0xFFFFFFFF, key_r, 31);
            unsigned int kk1_62 = _shfl_67;
            lose_r = lose_r | (int)(pk0_59 == packed) & ((int)(kk0_60 > key_r) | (int)(kk0_60 == key_r) & (int)(lane_0 > 15));
            lose_r = lose_r | (int)(pk1_61 == packed) & ((int)(kk1_62 > key_r) | (int)(kk1_62 == key_r) & (int)(lane_0 > 31));
            int packed_w = packed & ~lose_r;
            unsigned long long comp = (unsigned long long)key_r << 32 | (unsigned long long)((unsigned int)(2147483647 - packed_w) << 1 | (unsigned int)(1 - s_me));
            unsigned long long _shfl_68 = __shfl_sync(0xFFFFFFFF, comp, 47 - lane_0);
            unsigned long long rv = _shfl_68;
            if (lane_0 >= 16) {
                comp = rv;
            }
            unsigned long long _shfl_xor_176 = __shfl_xor_sync(0xFFFFFFFF, comp, 16);
            unsigned long long pv = _shfl_xor_176;
            int hi_l = (int)((lane_0 & 16) == 0);
            int sw = (int)(pv > comp) & hi_l | (int)(pv < comp) & 1 - hi_l;
            if (sw != 0) {
                comp = pv;
            }
            unsigned long long _shfl_xor_177 = __shfl_xor_sync(0xFFFFFFFF, comp, 8);
            unsigned long long pv_63 = _shfl_xor_177;
            int hi_l_64 = (int)((lane_0 & 8) == 0);
            int sw_65 = (int)(pv_63 > comp) & hi_l_64 | (int)(pv_63 < comp) & 1 - hi_l_64;
            if (sw_65 != 0) {
                comp = pv_63;
            }
            unsigned long long _shfl_xor_178 = __shfl_xor_sync(0xFFFFFFFF, comp, 4);
            unsigned long long pv_66 = _shfl_xor_178;
            int hi_l_67 = (int)((lane_0 & 4) == 0);
            int sw_68 = (int)(pv_66 > comp) & hi_l_67 | (int)(pv_66 < comp) & 1 - hi_l_67;
            if (sw_68 != 0) {
                comp = pv_66;
            }
            unsigned long long _shfl_xor_179 = __shfl_xor_sync(0xFFFFFFFF, comp, 2);
            unsigned long long pv_69 = _shfl_xor_179;
            int hi_l_70 = (int)((lane_0 & 2) == 0);
            int sw_71 = (int)(pv_69 > comp) & hi_l_70 | (int)(pv_69 < comp) & 1 - hi_l_70;
            if (sw_71 != 0) {
                comp = pv_69;
            }
            unsigned long long _shfl_xor_180 = __shfl_xor_sync(0xFFFFFFFF, comp, 1);
            unsigned long long pv_72 = _shfl_xor_180;
            int hi_l_73 = (int)((lane_0 & 1) == 0);
            int sw_74 = (int)(pv_72 > comp) & hi_l_73 | (int)(pv_72 < comp) & 1 - hi_l_73;
            if (sw_74 != 0) {
                comp = pv_72;
            }
            unsigned int low_s = (unsigned int)comp;
            int pw_s = 2147483647 - (int)(low_s >> 1);
            int win_s = pw_s & 1;
            int id_s = pw_s >> 1;
            unsigned int lmlt = (1 << (unsigned int)lane_0) - 1;
            unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, win_s != 0);
            unsigned int wm = _vote_3;
            int _popc_1 = __popc(wm & lmlt);
            int ordv = _popc_1;
            int _popc_2 = __popc(wm);
            int n_tot = _popc_2;
            int n_sel = ((n_tot < n_ord) ? n_tot : n_ord);
            int sid = ((win_s != 0 && ordv < n_ord) ? id_s : 2147483647);
            if (fb >= 0) {
                if (n_sel < 16) {
                    if (lane_0 == 31) {
                        sid = fb;
                    }
                } else if (lane_0 == 0) {
                    status[0] = 16;
                }
            }
            int _shfl_xor_181 = __shfl_xor_sync(0xFFFFFFFF, sid, 1);
            int pv_i = _shfl_xor_181;
            int asc = (int)((lane_0 & 2) == 0);
            int lower_j = (int)((lane_0 & 1) == 0);
            int keep_min = (int)(asc == lower_j);
            int sw2 = (int)(pv_i < sid) & keep_min | (int)(pv_i > sid) & 1 - keep_min;
            if (sw2 != 0) {
                sid = pv_i;
            }
            int _shfl_xor_182 = __shfl_xor_sync(0xFFFFFFFF, sid, 2);
            int pv_i_75 = _shfl_xor_182;
            int asc_76 = (int)((lane_0 & 4) == 0);
            int lower_j_77 = (int)((lane_0 & 2) == 0);
            int keep_min_78 = (int)(asc_76 == lower_j_77);
            int sw2_79 = (int)(pv_i_75 < sid) & keep_min_78 | (int)(pv_i_75 > sid) & 1 - keep_min_78;
            if (sw2_79 != 0) {
                sid = pv_i_75;
            }
            int _shfl_xor_183 = __shfl_xor_sync(0xFFFFFFFF, sid, 1);
            int pv_i_80 = _shfl_xor_183;
            int asc_81 = (int)((lane_0 & 4) == 0);
            int lower_j_82 = (int)((lane_0 & 1) == 0);
            int keep_min_83 = (int)(asc_81 == lower_j_82);
            int sw2_84 = (int)(pv_i_80 < sid) & keep_min_83 | (int)(pv_i_80 > sid) & 1 - keep_min_83;
            if (sw2_84 != 0) {
                sid = pv_i_80;
            }
            int _shfl_xor_184 = __shfl_xor_sync(0xFFFFFFFF, sid, 4);
            int pv_i_85 = _shfl_xor_184;
            int asc_86 = (int)((lane_0 & 8) == 0);
            int lower_j_87 = (int)((lane_0 & 4) == 0);
            int keep_min_88 = (int)(asc_86 == lower_j_87);
            int sw2_89 = (int)(pv_i_85 < sid) & keep_min_88 | (int)(pv_i_85 > sid) & 1 - keep_min_88;
            if (sw2_89 != 0) {
                sid = pv_i_85;
            }
            int _shfl_xor_185 = __shfl_xor_sync(0xFFFFFFFF, sid, 2);
            int pv_i_90 = _shfl_xor_185;
            int asc_91 = (int)((lane_0 & 8) == 0);
            int lower_j_92 = (int)((lane_0 & 2) == 0);
            int keep_min_93 = (int)(asc_91 == lower_j_92);
            int sw2_94 = (int)(pv_i_90 < sid) & keep_min_93 | (int)(pv_i_90 > sid) & 1 - keep_min_93;
            if (sw2_94 != 0) {
                sid = pv_i_90;
            }
            int _shfl_xor_186 = __shfl_xor_sync(0xFFFFFFFF, sid, 1);
            int pv_i_95 = _shfl_xor_186;
            int asc_96 = (int)((lane_0 & 8) == 0);
            int lower_j_97 = (int)((lane_0 & 1) == 0);
            int keep_min_98 = (int)(asc_96 == lower_j_97);
            int sw2_99 = (int)(pv_i_95 < sid) & keep_min_98 | (int)(pv_i_95 > sid) & 1 - keep_min_98;
            if (sw2_99 != 0) {
                sid = pv_i_95;
            }
            int _shfl_xor_187 = __shfl_xor_sync(0xFFFFFFFF, sid, 8);
            int pv_i_100 = _shfl_xor_187;
            int asc_101 = (int)((lane_0 & 16) == 0);
            int lower_j_102 = (int)((lane_0 & 8) == 0);
            int keep_min_103 = (int)(asc_101 == lower_j_102);
            int sw2_104 = (int)(pv_i_100 < sid) & keep_min_103 | (int)(pv_i_100 > sid) & 1 - keep_min_103;
            if (sw2_104 != 0) {
                sid = pv_i_100;
            }
            int _shfl_xor_188 = __shfl_xor_sync(0xFFFFFFFF, sid, 4);
            int pv_i_105 = _shfl_xor_188;
            int asc_106 = (int)((lane_0 & 16) == 0);
            int lower_j_107 = (int)((lane_0 & 4) == 0);
            int keep_min_108 = (int)(asc_106 == lower_j_107);
            int sw2_109 = (int)(pv_i_105 < sid) & keep_min_108 | (int)(pv_i_105 > sid) & 1 - keep_min_108;
            if (sw2_109 != 0) {
                sid = pv_i_105;
            }
            int _shfl_xor_189 = __shfl_xor_sync(0xFFFFFFFF, sid, 2);
            int pv_i_110 = _shfl_xor_189;
            int asc_111 = (int)((lane_0 & 16) == 0);
            int lower_j_112 = (int)((lane_0 & 2) == 0);
            int keep_min_113 = (int)(asc_111 == lower_j_112);
            int sw2_114 = (int)(pv_i_110 < sid) & keep_min_113 | (int)(pv_i_110 > sid) & 1 - keep_min_113;
            if (sw2_114 != 0) {
                sid = pv_i_110;
            }
            int _shfl_xor_190 = __shfl_xor_sync(0xFFFFFFFF, sid, 1);
            int pv_i_115 = _shfl_xor_190;
            int asc_116 = (int)((lane_0 & 16) == 0);
            int lower_j_117 = (int)((lane_0 & 1) == 0);
            int keep_min_118 = (int)(asc_116 == lower_j_117);
            int sw2_119 = (int)(pv_i_115 < sid) & keep_min_118 | (int)(pv_i_115 > sid) & 1 - keep_min_118;
            if (sw2_119 != 0) {
                sid = pv_i_115;
            }
            int _shfl_xor_191 = __shfl_xor_sync(0xFFFFFFFF, sid, 16);
            int pv_i_120 = _shfl_xor_191;
            int asc_121 = (int)((lane_0 & 32) == 0);
            int lower_j_122 = (int)((lane_0 & 16) == 0);
            int keep_min_123 = (int)(asc_121 == lower_j_122);
            int sw2_124 = (int)(pv_i_120 < sid) & keep_min_123 | (int)(pv_i_120 > sid) & 1 - keep_min_123;
            if (sw2_124 != 0) {
                sid = pv_i_120;
            }
            int _shfl_xor_192 = __shfl_xor_sync(0xFFFFFFFF, sid, 8);
            int pv_i_125 = _shfl_xor_192;
            int asc_126 = (int)((lane_0 & 32) == 0);
            int lower_j_127 = (int)((lane_0 & 8) == 0);
            int keep_min_128 = (int)(asc_126 == lower_j_127);
            int sw2_129 = (int)(pv_i_125 < sid) & keep_min_128 | (int)(pv_i_125 > sid) & 1 - keep_min_128;
            if (sw2_129 != 0) {
                sid = pv_i_125;
            }
            int _shfl_xor_193 = __shfl_xor_sync(0xFFFFFFFF, sid, 4);
            int pv_i_130 = _shfl_xor_193;
            int asc_131 = (int)((lane_0 & 32) == 0);
            int lower_j_132 = (int)((lane_0 & 4) == 0);
            int keep_min_133 = (int)(asc_131 == lower_j_132);
            int sw2_134 = (int)(pv_i_130 < sid) & keep_min_133 | (int)(pv_i_130 > sid) & 1 - keep_min_133;
            if (sw2_134 != 0) {
                sid = pv_i_130;
            }
            int _shfl_xor_194 = __shfl_xor_sync(0xFFFFFFFF, sid, 2);
            int pv_i_135 = _shfl_xor_194;
            int asc_136 = (int)((lane_0 & 32) == 0);
            int lower_j_137 = (int)((lane_0 & 2) == 0);
            int keep_min_138 = (int)(asc_136 == lower_j_137);
            int sw2_139 = (int)(pv_i_135 < sid) & keep_min_138 | (int)(pv_i_135 > sid) & 1 - keep_min_138;
            if (sw2_139 != 0) {
                sid = pv_i_135;
            }
            int _shfl_xor_195 = __shfl_xor_sync(0xFFFFFFFF, sid, 1);
            int pv_i_140 = _shfl_xor_195;
            int asc_141 = (int)((lane_0 & 32) == 0);
            int lower_j_142 = (int)((lane_0 & 1) == 0);
            int keep_min_143 = (int)(asc_141 == lower_j_142);
            int sw2_144 = (int)(pv_i_140 < sid) & keep_min_143 | (int)(pv_i_140 > sid) & 1 - keep_min_143;
            if (sw2_144 != 0) {
                sid = pv_i_140;
            }
            if (lane_0 < 16) {
                if (sid != 2147483647) {
                    out[(long long)row_x * 16 + (long long)lane_0] = sid;
                } else {
                    out[(long long)row_x * 16 + (long long)lane_0] = -1;
                }
            }
        }
        if (lane_0 == 0) {
            unsigned long long gtimer_2_2;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gtimer_2_2) :: "memory");
            unsigned long long t_endx = gtimer_2_2;
            long long sb = (long long)row * 3;
            stamps[sb] = t0;
            stamps[sb + 1] = t_ready;
            stamps[sb + 2] = t_endx;
            if (x_phase != 1) {
                gen_ctr[grow] = (g + 1) % 4094;
            }
        }
    }
}

} // extern "C"
