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

#include "cake_dsa_h64_train_device.cuh"

#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_nope_train_db791110b21a91be1388(int* __restrict__ indices, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ out, int* __restrict__ topk_length, int num_rows, int topk, int idx_stride, int indices_offset, int num_docs, int search_iters, int causal, int has_docs, int flat_bound, int write_out, int compact, int* __restrict__ length_in, int has_length_in)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    __shared__ __align__(16) unsigned char smem_static_raw[8320];

    // Kernel setup ops
    int* stage = reinterpret_cast<int*>(smem_static_raw + 0);
    const int stage_addr = (int)(unsigned long long)__cvta_generic_to_shared(stage);

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int t = blockIdx.x * 4 + warp;
    int lane_0 = lane;
    if (t < num_rows) {
        int key_base = 0;
        int bound = flat_bound;
        if (has_docs != 0) {
            int lo = 0;
            int hi = num_docs;
            #pragma unroll 1
            for (int it = 0; it < search_iters; it++) {
                int mid = (lo + hi) / 2;
                int _vec_load_0[1];
                {
                    _vec_load_0[0] = *reinterpret_cast<const int*>(cu_seqlens_q + (long long)mid + 0);
                }
                int go_lo = (int)(mid > lo && _vec_load_0[0] <= t);
                int go_hi = (int)(mid > lo && _vec_load_0[0] > t);
                lo = ((go_lo != 0) ? mid : lo);
                hi = ((go_hi != 0) ? mid : hi);
            }
            int _vec_load_1[1];
            {
                _vec_load_1[0] = *reinterpret_cast<const int*>(cu_seqlens_k + (long long)lo + 0);
            }
            int _vec_load_2[1];
            {
                _vec_load_2[0] = *reinterpret_cast<const int*>(cu_seqlens_k + (long long)(lo + 1) + 0);
            }
            key_base = _vec_load_1[0];
            int k_end = _vec_load_2[0];
            bound = k_end - key_base - 1;
            if (causal != 0) {
                int _vec_load_3[1];
                {
                    _vec_load_3[0] = *reinterpret_cast<const int*>(cu_seqlens_q + (long long)(lo + 1) + 0);
                }
                bound = k_end - _vec_load_3[0] + t - key_base;
            }
        }
        long long row_base = (long long)indices_offset + (long long)t * (long long)idx_stride;
        long long out_base = (long long)t * (long long)topk;
        int aligned = (int)(((idx_stride | indices_offset | topk) & 3) == 0);
        int compacting = (int)(compact != 0 && write_out != 0);
        if (compacting != 0) {
            int active_in = topk;
            if (has_length_in != 0) {
                int _max_1 = ((length_in[(long long)t]) > (0) ? (length_in[(long long)t]) : (0));
                int _min_0 = ((_max_1) < (topk) ? (_max_1) : (topk));
                active_in = _min_0;
            }
            int out_vec = (int)((topk & 3) == 0);
            int wbase = warp * 2 * 260;
            int par = 0;
            int count = 0;
            #pragma unroll 1
            for (int sblk = 0; sblk < (active_in + 255) / 256; sblk++) {
                int spos = sblk * 256 + lane_0 * 8;
                int whole = (int)(spos + 8 <= topk) & aligned;
                int vals[8];
                if (whole != 0) {
                    int _vec_load_4[4];
                    {
                        const int4* _ivptr_2 = reinterpret_cast<const int4*>(indices + (row_base + (long long)spos) + 0);
                        int4 _ivld_2;
                        _ivld_2 = *_ivptr_2;
                        _vec_load_4[0 + 0] = _ivld_2.x;
                        _vec_load_4[0 + 1] = _ivld_2.y;
                        _vec_load_4[0 + 2] = _ivld_2.z;
                        _vec_load_4[0 + 3] = _ivld_2.w;
                    }
                    int _vec_load_5[4];
                    {
                        const int4* _ivptr_3 = reinterpret_cast<const int4*>(indices + (row_base + (long long)spos + 4) + 0);
                        int4 _ivld_3;
                        _ivld_3 = *_ivptr_3;
                        _vec_load_5[0 + 0] = _ivld_3.x;
                        _vec_load_5[0 + 1] = _ivld_3.y;
                        _vec_load_5[0 + 2] = _ivld_3.z;
                        _vec_load_5[0 + 3] = _ivld_3.w;
                    }
                    vals[0] = _vec_load_4[0];
                    vals[4] = _vec_load_5[0];
                    vals[1] = _vec_load_4[1];
                    vals[5] = _vec_load_5[1];
                    vals[2] = _vec_load_4[2];
                    vals[6] = _vec_load_5[2];
                    vals[3] = _vec_load_4[3];
                    vals[7] = _vec_load_5[3];
                } else {
                    int sv = -1;
                    if (spos < topk) {
                        sv = indices[row_base + (long long)spos];
                    }
                    vals[0] = sv;
                    int sv_0 = -1;
                    if (spos + 1 < topk) {
                        sv_0 = indices[row_base + (long long)(spos + 1)];
                    }
                    vals[1] = sv_0;
                    int sv_1 = -1;
                    if (spos + 2 < topk) {
                        sv_1 = indices[row_base + (long long)(spos + 2)];
                    }
                    vals[2] = sv_1;
                    int sv_2 = -1;
                    if (spos + 3 < topk) {
                        sv_2 = indices[row_base + (long long)(spos + 3)];
                    }
                    vals[3] = sv_2;
                    int sv_3 = -1;
                    if (spos + 4 < topk) {
                        sv_3 = indices[row_base + (long long)(spos + 4)];
                    }
                    vals[4] = sv_3;
                    int sv_4 = -1;
                    if (spos + 5 < topk) {
                        sv_4 = indices[row_base + (long long)(spos + 5)];
                    }
                    vals[5] = sv_4;
                    int sv_5 = -1;
                    if (spos + 6 < topk) {
                        sv_5 = indices[row_base + (long long)(spos + 6)];
                    }
                    vals[6] = sv_5;
                    int sv_6 = -1;
                    if (spos + 7 < topk) {
                        sv_6 = indices[row_base + (long long)(spos + 7)];
                    }
                    vals[7] = sv_6;
                }
                int outv[8];
                int ok[8];
                unsigned int hit = 0;
                int v = vals[0];
                ok[0] = (int)(v >= 0 && v <= bound && active_in > spos);
                outv[0] = v + key_base;
                hit = hit + (unsigned int)ok[0];
                int v_0 = vals[1];
                ok[1] = (int)(v_0 >= 0 && v_0 <= bound && active_in > spos + 1);
                outv[1] = v_0 + key_base;
                hit = hit + (unsigned int)ok[1];
                int v_1 = vals[2];
                ok[2] = (int)(v_1 >= 0 && v_1 <= bound && active_in > spos + 2);
                outv[2] = v_1 + key_base;
                hit = hit + (unsigned int)ok[2];
                int v_2 = vals[3];
                ok[3] = (int)(v_2 >= 0 && v_2 <= bound && active_in > spos + 3);
                outv[3] = v_2 + key_base;
                hit = hit + (unsigned int)ok[3];
                int v_3 = vals[4];
                ok[4] = (int)(v_3 >= 0 && v_3 <= bound && active_in > spos + 4);
                outv[4] = v_3 + key_base;
                hit = hit + (unsigned int)ok[4];
                int v_4 = vals[5];
                ok[5] = (int)(v_4 >= 0 && v_4 <= bound && active_in > spos + 5);
                outv[5] = v_4 + key_base;
                hit = hit + (unsigned int)ok[5];
                int v_5 = vals[6];
                ok[6] = (int)(v_5 >= 0 && v_5 <= bound && active_in > spos + 6);
                outv[6] = v_5 + key_base;
                hit = hit + (unsigned int)ok[6];
                int v_6 = vals[7];
                ok[7] = (int)(v_6 >= 0 && v_6 <= bound && active_in > spos + 7);
                outv[7] = v_6 + key_base;
                hit = hit + (unsigned int)ok[7];
                uint32_t incl = hit;
                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(incl) : "r"(1));
                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(incl) : "r"(2));
                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(incl) : "r"(4));
                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(incl) : "r"(8));
                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(incl) : "r"(16));
                unsigned int _shfl_sync_4 = __shfl_sync(0xFFFFFFFF, incl, 31);
                int tot = (int)_shfl_sync_4;
                if (tot == 256 && aligned != 0 && (count & 3) == 0) {
                    {
                        int4 _iv4 = make_int4(outv[0 + 0], outv[0 + 1], outv[0 + 2], outv[0 + 3]);
                        *reinterpret_cast<int4*>(out + out_base + (long long)(count + lane_0 * 8)) = _iv4;
                    }
                    {
                        int4 _iv4 = make_int4(outv[4 + 0], outv[4 + 1], outv[4 + 2], outv[4 + 3]);
                        *reinterpret_cast<int4*>(out + out_base + (long long)(count + lane_0 * 8) + 4) = _iv4;
                    }
                } else {
                    int pend = (count & 3) * out_vec;
                    int wbuf = wbase + par * 260;
                    int wpos = wbuf + pend + (int)(incl - hit);
                    if (ok[0] != 0) {
                        stage[wpos] = outv[0];
                        wpos = wpos + 1;
                    }
                    if (ok[1] != 0) {
                        stage[wpos] = outv[1];
                        wpos = wpos + 1;
                    }
                    if (ok[2] != 0) {
                        stage[wpos] = outv[2];
                        wpos = wpos + 1;
                    }
                    if (ok[3] != 0) {
                        stage[wpos] = outv[3];
                        wpos = wpos + 1;
                    }
                    if (ok[4] != 0) {
                        stage[wpos] = outv[4];
                        wpos = wpos + 1;
                    }
                    if (ok[5] != 0) {
                        stage[wpos] = outv[5];
                        wpos = wpos + 1;
                    }
                    if (ok[6] != 0) {
                        stage[wpos] = outv[6];
                        wpos = wpos + 1;
                    }
                    if (ok[7] != 0) {
                        stage[wpos] = outv[7];
                        wpos = wpos + 1;
                    }
                    __syncwarp();
                    if (out_vec != 0) {
                        int qb = (pend + tot) / 4 * 4;
                        long long dst = out_base + (long long)(count - pend);
                        int q = lane_0;
                        if (qb > q * 4) {
                            int quad[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&quad[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 3]))
                                : "r"(stage_addr + (unsigned int)((wbuf + q * 4) * 4)));
                            {
                                int4 _iv4 = make_int4(quad[0 + 0], quad[0 + 1], quad[0 + 2], quad[0 + 3]);
                                *reinterpret_cast<int4*>(out + dst + (long long)(q * 4)) = _iv4;
                            }
                        }
                        int q_0 = lane_0 + 32;
                        if (qb > q_0 * 4) {
                            int quad_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&quad_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad_1[(0) + 3]))
                                : "r"(stage_addr + (unsigned int)((wbuf + q_0 * 4) * 4)));
                            {
                                int4 _iv4 = make_int4(quad_1[0 + 0], quad_1[0 + 1], quad_1[0 + 2], quad_1[0 + 3]);
                                *reinterpret_cast<int4*>(out + dst + (long long)(q_0 * 4)) = _iv4;
                            }
                        }
                        if (lane_0 < pend + tot - qb) {
                            int carried = stage[wbuf + qb + lane_0];
                            stage[wbase + (1 - par) * 260 + lane_0] = carried;
                        }
                    } else {
                        #pragma unroll 1
                        for (int k = 0; k < (tot + 31) / 32; k++) {
                            int r = lane_0 + k * 32;
                            if (r < tot) {
                                int w = stage[wbuf + r];
                                out[out_base + (long long)(count + r)] = w;
                            }
                        }
                    }
                    par = 1 - par;
                }
                count = count + tot;
            }
            if (out_vec != 0) {
                int pend_1 = count & 3;
                int wbuf_1 = wbase + par * 260;
                if (lane_0 < 4) {
                    int g = count - pend_1 + lane_0;
                    int w_1 = -1;
                    if (lane_0 < pend_1) {
                        w_1 = stage[wbuf_1 + lane_0];
                    }
                    if (g < topk) {
                        out[out_base + (long long)g] = w_1;
                    }
                }
                int neg4[4];
                neg4[0] = -1;
                neg4[1] = -1;
                neg4[2] = -1;
                neg4[3] = -1;
                int start = count - pend_1 + 4;
                int nq_tail = (topk - start) / 4;
                #pragma unroll 1
                for (int k_1 = 0; k_1 < (nq_tail + 31) / 32; k_1++) {
                    int q_1 = lane_0 + k_1 * 32;
                    if (q_1 < nq_tail) {
                        {
                            int4 _iv4 = make_int4(neg4[0 + 0], neg4[0 + 1], neg4[0 + 2], neg4[0 + 3]);
                            *reinterpret_cast<int4*>(out + out_base + (long long)(start + q_1 * 4)) = _iv4;
                        }
                    }
                }
            } else {
                #pragma unroll 1
                for (int k_2 = 0; k_2 < (topk - count + 31) / 32; k_2++) {
                    int p = count + lane_0 + k_2 * 32;
                    if (p < topk) {
                        out[out_base + (long long)p] = -1;
                    }
                }
            }
            if (lane_0 == 0) {
                topk_length[(long long)t] = count;
            }
        } else {
            int last = 0;
            #pragma unroll 1
            for (int sblk_1 = 0; sblk_1 < (topk + 255) / 256; sblk_1++) {
                int spos_1 = sblk_1 * 256 + lane_0 * 8;
                int whole_1 = (int)(spos_1 + 8 <= topk) & aligned;
                int vals_1[8];
                if (whole_1 != 0) {
                    int _vec_load_6[4];
                    {
                        const int4* _ivptr_5 = reinterpret_cast<const int4*>(indices + (row_base + (long long)spos_1) + 0);
                        int4 _ivld_5;
                        _ivld_5 = *_ivptr_5;
                        _vec_load_6[0 + 0] = _ivld_5.x;
                        _vec_load_6[0 + 1] = _ivld_5.y;
                        _vec_load_6[0 + 2] = _ivld_5.z;
                        _vec_load_6[0 + 3] = _ivld_5.w;
                    }
                    int _vec_load_7[4];
                    {
                        const int4* _ivptr_6 = reinterpret_cast<const int4*>(indices + (row_base + (long long)spos_1 + 4) + 0);
                        int4 _ivld_6;
                        _ivld_6 = *_ivptr_6;
                        _vec_load_7[0 + 0] = _ivld_6.x;
                        _vec_load_7[0 + 1] = _ivld_6.y;
                        _vec_load_7[0 + 2] = _ivld_6.z;
                        _vec_load_7[0 + 3] = _ivld_6.w;
                    }
                    vals_1[0] = _vec_load_6[0];
                    vals_1[4] = _vec_load_7[0];
                    vals_1[1] = _vec_load_6[1];
                    vals_1[5] = _vec_load_7[1];
                    vals_1[2] = _vec_load_6[2];
                    vals_1[6] = _vec_load_7[2];
                    vals_1[3] = _vec_load_6[3];
                    vals_1[7] = _vec_load_7[3];
                } else {
                    int sv_7 = -1;
                    if (spos_1 < topk) {
                        sv_7 = indices[row_base + (long long)spos_1];
                    }
                    vals_1[0] = sv_7;
                    int sv_0_1 = -1;
                    if (spos_1 + 1 < topk) {
                        sv_0_1 = indices[row_base + (long long)(spos_1 + 1)];
                    }
                    vals_1[1] = sv_0_1;
                    int sv_1_1 = -1;
                    if (spos_1 + 2 < topk) {
                        sv_1_1 = indices[row_base + (long long)(spos_1 + 2)];
                    }
                    vals_1[2] = sv_1_1;
                    int sv_2_1 = -1;
                    if (spos_1 + 3 < topk) {
                        sv_2_1 = indices[row_base + (long long)(spos_1 + 3)];
                    }
                    vals_1[3] = sv_2_1;
                    int sv_3_1 = -1;
                    if (spos_1 + 4 < topk) {
                        sv_3_1 = indices[row_base + (long long)(spos_1 + 4)];
                    }
                    vals_1[4] = sv_3_1;
                    int sv_4_1 = -1;
                    if (spos_1 + 5 < topk) {
                        sv_4_1 = indices[row_base + (long long)(spos_1 + 5)];
                    }
                    vals_1[5] = sv_4_1;
                    int sv_5_1 = -1;
                    if (spos_1 + 6 < topk) {
                        sv_5_1 = indices[row_base + (long long)(spos_1 + 6)];
                    }
                    vals_1[6] = sv_5_1;
                    int sv_6_1 = -1;
                    if (spos_1 + 7 < topk) {
                        sv_6_1 = indices[row_base + (long long)(spos_1 + 7)];
                    }
                    vals_1[7] = sv_6_1;
                }
                int outv_1[8];
                int v_7 = vals_1[0];
                int ok_1 = (int)(v_7 >= 0 && v_7 <= bound);
                outv_1[0] = -1;
                if (ok_1 != 0) {
                    outv_1[0] = v_7 + key_base;
                    last = spos_1 + 1;
                }
                int v_0_1 = vals_1[1];
                int ok_1_1 = (int)(v_0_1 >= 0 && v_0_1 <= bound);
                outv_1[1] = -1;
                if (ok_1_1 != 0) {
                    outv_1[1] = v_0_1 + key_base;
                    last = spos_1 + 1 + 1;
                }
                int v_2_1 = vals_1[2];
                int ok_3 = (int)(v_2_1 >= 0 && v_2_1 <= bound);
                outv_1[2] = -1;
                if (ok_3 != 0) {
                    outv_1[2] = v_2_1 + key_base;
                    last = spos_1 + 2 + 1;
                }
                int v_4_1 = vals_1[3];
                int ok_5 = (int)(v_4_1 >= 0 && v_4_1 <= bound);
                outv_1[3] = -1;
                if (ok_5 != 0) {
                    outv_1[3] = v_4_1 + key_base;
                    last = spos_1 + 3 + 1;
                }
                int v_6_1 = vals_1[4];
                int ok_7 = (int)(v_6_1 >= 0 && v_6_1 <= bound);
                outv_1[4] = -1;
                if (ok_7 != 0) {
                    outv_1[4] = v_6_1 + key_base;
                    last = spos_1 + 4 + 1;
                }
                int v_8 = vals_1[5];
                int ok_9 = (int)(v_8 >= 0 && v_8 <= bound);
                outv_1[5] = -1;
                if (ok_9 != 0) {
                    outv_1[5] = v_8 + key_base;
                    last = spos_1 + 5 + 1;
                }
                int v_10 = vals_1[6];
                int ok_11 = (int)(v_10 >= 0 && v_10 <= bound);
                outv_1[6] = -1;
                if (ok_11 != 0) {
                    outv_1[6] = v_10 + key_base;
                    last = spos_1 + 6 + 1;
                }
                int v_12 = vals_1[7];
                int ok_13 = (int)(v_12 >= 0 && v_12 <= bound);
                outv_1[7] = -1;
                if (ok_13 != 0) {
                    outv_1[7] = v_12 + key_base;
                    last = spos_1 + 7 + 1;
                }
                if (write_out != 0) {
                    if (whole_1 != 0) {
                        {
                            int4 _iv4 = make_int4(outv_1[0 + 0], outv_1[0 + 1], outv_1[0 + 2], outv_1[0 + 3]);
                            *reinterpret_cast<int4*>(out + out_base + (long long)spos_1) = _iv4;
                        }
                        {
                            int4 _iv4 = make_int4(outv_1[4 + 0], outv_1[4 + 1], outv_1[4 + 2], outv_1[4 + 3]);
                            *reinterpret_cast<int4*>(out + out_base + (long long)spos_1 + 4) = _iv4;
                        }
                    } else {
                        if (spos_1 < topk) {
                            out[out_base + (long long)spos_1] = outv_1[0];
                        }
                        if (spos_1 + 1 < topk) {
                            out[out_base + (long long)(spos_1 + 1)] = outv_1[1];
                        }
                        if (spos_1 + 2 < topk) {
                            out[out_base + (long long)(spos_1 + 2)] = outv_1[2];
                        }
                        if (spos_1 + 3 < topk) {
                            out[out_base + (long long)(spos_1 + 3)] = outv_1[3];
                        }
                        if (spos_1 + 4 < topk) {
                            out[out_base + (long long)(spos_1 + 4)] = outv_1[4];
                        }
                        if (spos_1 + 5 < topk) {
                            out[out_base + (long long)(spos_1 + 5)] = outv_1[5];
                        }
                        if (spos_1 + 6 < topk) {
                            out[out_base + (long long)(spos_1 + 6)] = outv_1[6];
                        }
                        if (spos_1 + 7 < topk) {
                            out[out_base + (long long)(spos_1 + 7)] = outv_1[7];
                        }
                    }
                }
            }
            int row_len;
            asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(row_len) : "r"(last));
            if (lane_0 == 0) {
                topk_length[(long long)t] = row_len;
            }
        }
    }
}

} // extern "C"
