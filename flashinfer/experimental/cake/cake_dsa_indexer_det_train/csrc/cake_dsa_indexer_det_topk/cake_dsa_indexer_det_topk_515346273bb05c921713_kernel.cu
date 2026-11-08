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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_dsa_indexer_det_topk_515346273bb05c921713(long long* __restrict__ Staging, int* __restrict__ Indices, float* __restrict__ Scores, int top_k, int n_split)
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

    // Prologue
    int* s_m = reinterpret_cast<int*>(smem_raw + 0);
    const int s_m_addr = smem + 0;
    unsigned int* s_k = reinterpret_cast<unsigned int*>(smem_raw + 17664);
    const int s_k_addr = smem + 17664;
    int* s_kid = reinterpret_cast<int*>(smem_raw + 0);
    const int s_kid_addr = smem + 0;
    unsigned int* s_kb = reinterpret_cast<unsigned int*>(smem_raw + 8192);
    const int s_kb_addr = smem + 8192;
    unsigned int* s_xr = reinterpret_cast<unsigned int*>(smem_raw + 16384);
    const int s_xr_addr = smem + 16384;
    unsigned int* s_bm = reinterpret_cast<unsigned int*>(smem_raw + 214272);
    const int s_bm_addr = smem + 214272;
    uint16_t* s_pref = reinterpret_cast<uint16_t*>(smem_raw + 17664);
    const int s_pref_addr = smem + 17664;
    int* s_oid = reinterpret_cast<int*>(smem_raw + 18176);
    const int s_oid_addr = smem + 18176;
    unsigned int* s_osc = reinterpret_cast<unsigned int*>(smem_raw + 26368);
    const int s_osc_addr = smem + 26368;

    // === Task calls (declaration order) ===
    int lane_1 = tid % 32;
    int warp_1 = tid / 32;
    int n = n_split * top_k;
    int n_pad = (n + 8191) / 8192 * 8192;
    int res = ((n_pad <= 49152) ? 1 : 0);
    int n_r = ((n_pad) < (49152) ? (n_pad) : (49152));
    long long base = (long long)bid * (long long)n;
    long long row_base = (long long)bid * (long long)top_k;
    unsigned int zero_u = 0;
    unsigned int sign_mask = 2147483648ll;
    unsigned int nan_floor = 2139095040;
    unsigned int key_pad = 1;
    int hist_base = warp_1 * 256;
    int sink_idx = 4352 + lane_1;
    int k_rem = top_k;
    int kept_above = 0;
    unsigned long long prefix = 0;
    unsigned long long edge = 0;
    int c_valid = 0;
    int my_count = 0;
    int own_valid = 0;
    unsigned int one_r = 1;
    if (tid < 256) {
        s_bm[tid] = zero_u;
    }
    #pragma unroll 1
    for (int p = 0; p < 8; p++) {
        int shift = 56 - 8 * p;
        s_m[hist_base + lane_1 * 8] = 0;
        s_m[hist_base + lane_1 * 8 + 1] = 0;
        s_m[hist_base + lane_1 * 8 + 2] = 0;
        s_m[hist_base + lane_1 * 8 + 3] = 0;
        s_m[hist_base + lane_1 * 8 + 4] = 0;
        s_m[hist_base + lane_1 * 8 + 5] = 0;
        s_m[hist_base + lane_1 * 8 + 6] = 0;
        s_m[hist_base + lane_1 * 8 + 7] = 0;
        __syncwarp();
        if (res == 0) {
            #pragma unroll 1
            for (int e0 = 0; e0 < n; e0 += 4096) {
                unsigned long long ent[8];
                #pragma unroll
                for (int u = 0; u < 8; u++) {
                    int idx_l = e0 + tid + 512 * u;
                    ent[u] = 4294967295ll;
                    if (idx_l < n) {
                        ent[u] = (unsigned long long)Staging[base + (long long)idx_l];
                    }
                }
                #pragma unroll
                for (int u_1 = 0; u_1 < 8; u_1++) {
                    int kid = (int)(unsigned int)(ent[u_1] & 4294967295ull);
                    int kid_h = kid;
                    int valid_h = ((kid_h != -1) ? 1 : 0);
                    unsigned int bits = (unsigned int)(ent[u_1] >> 32ull);
                    unsigned int magnitude = bits & 2147483647;
                    unsigned int m = ((magnitude != 0) ? bits : zero_u);
                    unsigned int key32 = (((m & sign_mask) != 0) ? ~m : m | sign_mask);
                    if (magnitude > nan_floor) {
                        key32 = zero_u;
                    }
                    unsigned long long key64 = (unsigned long long)key32 << 32ull | (unsigned long long)(unsigned int)kid_h;
                    unsigned long long key_h = key64;
                    unsigned long long ks_h = key_h >> (unsigned long long)shift;
                    int digit_h = (int)(ks_h & 255ull);
                    int pm_h = ((ks_h >> 8ull == prefix) ? 1 : 0);
                    int bin_h = (((valid_h & pm_h) != 0) ? hist_base + digit_h : sink_idx);
                    atomicAdd(&s_m[bin_h], 1);
                }
            }
        } else if (p == 0) {
            #pragma unroll 1
            for (int e0_1 = 0; e0_1 < n; e0_1 += 16384) {
                unsigned long long ld[32];
                #pragma unroll
                for (int u_2 = 0; u_2 < 32; u_2++) {
                    int idx_0 = e0_1 + tid + 512 * u_2;
                    ld[u_2] = 4294967295ll;
                    if (idx_0 < n) {
                        ld[u_2] = (unsigned long long)Staging[base + (long long)idx_0];
                    }
                }
                #pragma unroll
                for (int u_3 = 0; u_3 < 32; u_3++) {
                    int idx_s = e0_1 + tid + 512 * u_3;
                    int kid_1 = (int)(unsigned int)(ld[u_3] & 4294967295ull);
                    int kid_0 = kid_1;
                    unsigned int bits_1 = (unsigned int)(ld[u_3] >> 32ull);
                    unsigned int magnitude_1 = bits_1 & 2147483647;
                    unsigned int m_1 = ((magnitude_1 != 0) ? bits_1 : zero_u);
                    unsigned int key32_1 = (((m_1 & sign_mask) != 0) ? ~m_1 : m_1 | sign_mask);
                    if (magnitude_1 > nan_floor) {
                        key32_1 = zero_u;
                    }
                    unsigned int key_0 = ((kid_0 != -1) ? key32_1 : key_pad);
                    if (idx_s < n_r) {
                        s_k[idx_s] = key_0;
                    }
                    int bin_0 = ((kid_0 != -1) ? hist_base + (int)(key32_1 >> 24) : sink_idx);
                    atomicAdd(&s_m[bin_0], 1);
                }
            }
        } else {
            if (p < 4) {
                unsigned int sh = (unsigned int)(shift - 32);
                unsigned int pre32 = (unsigned int)prefix;
                int own_off = lane_1 / 8 * 512 + warp_1 * 32 + lane_1 % 8 * 4;
                #pragma unroll 1
                for (int b0 = 0; b0 < n_r / 512; b0 += 8) {
                    unsigned int w[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w[0])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 3]))
                        : "r"(s_k_addr + (unsigned int)((b0 * 512 + own_off) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w[4])), "=r"(*reinterpret_cast<uint32_t*>(&w[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w[(4) + 3]))
                        : "r"(s_k_addr + (unsigned int)((b0 * 512 + 2048 + own_off) * 4)));
                    #pragma unroll
                    for (int q = 0; q < 8; q++) {
                        unsigned int ks_s = w[q] >> sh;
                        if (ks_s >> 8 == pre32) {
                            atomicAdd(&s_m[hist_base + (int)(ks_s & 255)], 1);
                        }
                    }
                }
            } else {
                unsigned int t32 = (unsigned int)(prefix >> (unsigned long long)(8 * p - 32));
                #pragma unroll 1
                for (int e0_2 = 0; e0_2 < n_r; e0_2 += 4096) {
                    unsigned long long ent_t[8];
                    #pragma unroll
                    for (int u_4 = 0; u_4 < 8; u_4++) {
                        int idx_t = e0_2 + tid + 512 * u_4;
                        ent_t[u_4] = 4294967295ll;
                        if (s_k[idx_t] == t32) {
                            ent_t[u_4] = (unsigned long long)Staging[base + (long long)idx_t];
                        }
                    }
                    #pragma unroll
                    for (int u_5 = 0; u_5 < 8; u_5++) {
                        int kid_2 = (int)(unsigned int)(ent_t[u_5] & 4294967295ull);
                        int kid_t = kid_2;
                        if (kid_t != -1) {
                            unsigned long long key_t = (unsigned long long)t32 << 32ull | (unsigned long long)(unsigned int)kid_t;
                            unsigned long long ks_t = key_t >> (unsigned long long)shift;
                            if (ks_t >> 8ull == prefix) {
                                atomicAdd(&s_m[hist_base + (int)(ks_t & 255ull)], 1);
                            }
                        }
                    }
                }
            }
        }
        int pad_fix = ((res != 0 && p >= 1 && p < 4 && prefix == 0ull) ? 1 : 0);
        int pad_bin = ((p == 3) ? 1 : 0);
        __syncthreads();
        if (tid < 256) {
            int tot = 0;
            tot += s_m[tid];
            tot += s_m[256 + tid];
            tot += s_m[512 + tid];
            tot += s_m[768 + tid];
            tot += s_m[1024 + tid];
            tot += s_m[1280 + tid];
            tot += s_m[1536 + tid];
            tot += s_m[1792 + tid];
            tot += s_m[2048 + tid];
            tot += s_m[2304 + tid];
            tot += s_m[2560 + tid];
            tot += s_m[2816 + tid];
            tot += s_m[3072 + tid];
            tot += s_m[3328 + tid];
            tot += s_m[3584 + tid];
            tot += s_m[3840 + tid];
            if (pad_fix != 0 && tid == pad_bin) {
                tot -= n_pad - c_valid;
            }
            s_m[4096 + tid] = tot;
        }
        __syncthreads();
        int lane_bins[8];
        int own_bins[8];
        int lane_sum = 0;
        lane_bins[0] = s_m[4096 + lane_1 * 8];
        own_bins[0] = s_m[hist_base + lane_1 * 8];
        if (pad_fix != 0 && lane_1 * 8 == pad_bin) {
            own_bins[0] = own_bins[0] - (n_pad / 16 - own_valid);
        }
        lane_sum += lane_bins[0];
        lane_bins[1] = s_m[4096 + lane_1 * 8 + 1];
        own_bins[1] = s_m[hist_base + lane_1 * 8 + 1];
        if (pad_fix != 0 && lane_1 * 8 + 1 == pad_bin) {
            own_bins[1] = own_bins[1] - (n_pad / 16 - own_valid);
        }
        lane_sum += lane_bins[1];
        lane_bins[2] = s_m[4096 + lane_1 * 8 + 2];
        own_bins[2] = s_m[hist_base + lane_1 * 8 + 2];
        if (pad_fix != 0 && lane_1 * 8 + 2 == pad_bin) {
            own_bins[2] = own_bins[2] - (n_pad / 16 - own_valid);
        }
        lane_sum += lane_bins[2];
        lane_bins[3] = s_m[4096 + lane_1 * 8 + 3];
        own_bins[3] = s_m[hist_base + lane_1 * 8 + 3];
        if (pad_fix != 0 && lane_1 * 8 + 3 == pad_bin) {
            own_bins[3] = own_bins[3] - (n_pad / 16 - own_valid);
        }
        lane_sum += lane_bins[3];
        lane_bins[4] = s_m[4096 + lane_1 * 8 + 4];
        own_bins[4] = s_m[hist_base + lane_1 * 8 + 4];
        if (pad_fix != 0 && lane_1 * 8 + 4 == pad_bin) {
            own_bins[4] = own_bins[4] - (n_pad / 16 - own_valid);
        }
        lane_sum += lane_bins[4];
        lane_bins[5] = s_m[4096 + lane_1 * 8 + 5];
        own_bins[5] = s_m[hist_base + lane_1 * 8 + 5];
        if (pad_fix != 0 && lane_1 * 8 + 5 == pad_bin) {
            own_bins[5] = own_bins[5] - (n_pad / 16 - own_valid);
        }
        lane_sum += lane_bins[5];
        lane_bins[6] = s_m[4096 + lane_1 * 8 + 6];
        own_bins[6] = s_m[hist_base + lane_1 * 8 + 6];
        if (pad_fix != 0 && lane_1 * 8 + 6 == pad_bin) {
            own_bins[6] = own_bins[6] - (n_pad / 16 - own_valid);
        }
        lane_sum += lane_bins[6];
        lane_bins[7] = s_m[4096 + lane_1 * 8 + 7];
        own_bins[7] = s_m[hist_base + lane_1 * 8 + 7];
        if (pad_fix != 0 && lane_1 * 8 + 7 == pad_bin) {
            own_bins[7] = own_bins[7] - (n_pad / 16 - own_valid);
        }
        lane_sum += lane_bins[7];
        uint32_t _warp_scan_sum_0 = (unsigned int)lane_sum;
        { uint32_t peer = __shfl_down_sync(0xFFFFFFFF, _warp_scan_sum_0, 1, 32); if (lane + 1 < 32) { _warp_scan_sum_0 += peer; } }
        { uint32_t peer = __shfl_down_sync(0xFFFFFFFF, _warp_scan_sum_0, 2, 32); if (lane + 2 < 32) { _warp_scan_sum_0 += peer; } }
        { uint32_t peer = __shfl_down_sync(0xFFFFFFFF, _warp_scan_sum_0, 4, 32); if (lane + 4 < 32) { _warp_scan_sum_0 += peer; } }
        { uint32_t peer = __shfl_down_sync(0xFFFFFFFF, _warp_scan_sum_0, 8, 32); if (lane + 8 < 32) { _warp_scan_sum_0 += peer; } }
        { uint32_t peer = __shfl_down_sync(0xFFFFFFFF, _warp_scan_sum_0, 16, 32); if (lane + 16 < 32) { _warp_scan_sum_0 += peer; } }
        int suffix = (int)_warp_scan_sum_0;
        if (p == 0) {
            int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, suffix, 0);
            c_valid = _shfl_sync_1;
            int own_all = 0;
            own_all += own_bins[0];
            own_all += own_bins[1];
            own_all += own_bins[2];
            own_all += own_bins[3];
            own_all += own_bins[4];
            own_all += own_bins[5];
            own_all += own_bins[6];
            own_all += own_bins[7];
            int _warp_redux_2;
            asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_2) : "r"(own_all));
            own_valid = _warp_redux_2;
        }
        if (c_valid <= top_k) {
            my_count = own_valid;
            edge = 0;
            break;
        }
        int excl = suffix - lane_sum;
        int is_target = ((excl < k_rem && k_rem <= excl + lane_sum) ? 1 : 0);
        int d_sel = 0;
        int above_sel = 0;
        int count_sel = 0;
        int found = 0;
        int cum_above = excl;
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[7]) {
                d_sel = 7;
                above_sel = cum_above;
                count_sel = lane_bins[7];
                found = 1;
            }
        }
        cum_above += lane_bins[7];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[6]) {
                d_sel = 6;
                above_sel = cum_above;
                count_sel = lane_bins[6];
                found = 1;
            }
        }
        cum_above += lane_bins[6];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[5]) {
                d_sel = 5;
                above_sel = cum_above;
                count_sel = lane_bins[5];
                found = 1;
            }
        }
        cum_above += lane_bins[5];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[4]) {
                d_sel = 4;
                above_sel = cum_above;
                count_sel = lane_bins[4];
                found = 1;
            }
        }
        cum_above += lane_bins[4];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[3]) {
                d_sel = 3;
                above_sel = cum_above;
                count_sel = lane_bins[3];
                found = 1;
            }
        }
        cum_above += lane_bins[3];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[2]) {
                d_sel = 2;
                above_sel = cum_above;
                count_sel = lane_bins[2];
                found = 1;
            }
        }
        cum_above += lane_bins[2];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[1]) {
                d_sel = 1;
                above_sel = cum_above;
                count_sel = lane_bins[1];
                found = 1;
            }
        }
        cum_above += lane_bins[1];
        if (found == 0) {
            if (k_rem <= cum_above + lane_bins[0]) {
                d_sel = 0;
                above_sel = cum_above;
                count_sel = lane_bins[0];
                found = 1;
            }
        }
        cum_above += lane_bins[0];
        int target_lane;
        asm volatile("redux.sync.max.s32 %0, %1, 0xFFFFFFFF;" : "=r"(target_lane) : "r"(is_target * lane_1));
        int digit_sel = __shfl_sync(0xFFFFFFFF, d_sel, target_lane);
        int above_cnt = __shfl_sync(0xFFFFFFFF, above_sel, target_lane);
        int bucket_cnt = __shfl_sync(0xFFFFFFFF, count_sel, target_lane);
        k_rem = k_rem - above_cnt;
        kept_above += above_cnt;
        prefix = prefix << 8ull | (unsigned long long)(unsigned int)(target_lane * 8 + digit_sel);
        edge = prefix << (unsigned long long)shift;
        int sel_bin = target_lane * 8 + digit_sel;
        int last = ((kept_above + bucket_cnt <= top_k) ? 1 : 0);
        int own_kept = 0;
        if (lane_1 * 8 > sel_bin || lane_1 * 8 == sel_bin && last != 0) {
            own_kept += own_bins[0];
        }
        if (lane_1 * 8 + 1 > sel_bin || lane_1 * 8 + 1 == sel_bin && last != 0) {
            own_kept += own_bins[1];
        }
        if (lane_1 * 8 + 2 > sel_bin || lane_1 * 8 + 2 == sel_bin && last != 0) {
            own_kept += own_bins[2];
        }
        if (lane_1 * 8 + 3 > sel_bin || lane_1 * 8 + 3 == sel_bin && last != 0) {
            own_kept += own_bins[3];
        }
        if (lane_1 * 8 + 4 > sel_bin || lane_1 * 8 + 4 == sel_bin && last != 0) {
            own_kept += own_bins[4];
        }
        if (lane_1 * 8 + 5 > sel_bin || lane_1 * 8 + 5 == sel_bin && last != 0) {
            own_kept += own_bins[5];
        }
        if (lane_1 * 8 + 6 > sel_bin || lane_1 * 8 + 6 == sel_bin && last != 0) {
            own_kept += own_bins[6];
        }
        if (lane_1 * 8 + 7 > sel_bin || lane_1 * 8 + 7 == sel_bin && last != 0) {
            own_kept += own_bins[7];
        }
        int _warp_redux_3;
        asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_3) : "r"(own_kept));
        my_count += _warp_redux_3;
        if (last != 0) {
            break;
        }
    }
    if (lane_1 == 0) {
        s_m[4384 + warp_1] = my_count;
    }
    __syncthreads();
    int write_pos = 0;
    int n_kept = 0;
    int cnt_w_15_0 = s_m[4384];
    n_kept += cnt_w_15_0;
    write_pos += ((warp_1 > 0) ? cnt_w_15_0 : 0);
    int cnt_w_15_1 = s_m[4385];
    n_kept += cnt_w_15_1;
    write_pos += ((warp_1 > 1) ? cnt_w_15_1 : 0);
    int cnt_w_15_2 = s_m[4386];
    n_kept += cnt_w_15_2;
    write_pos += ((warp_1 > 2) ? cnt_w_15_2 : 0);
    int cnt_w_15_3 = s_m[4387];
    n_kept += cnt_w_15_3;
    write_pos += ((warp_1 > 3) ? cnt_w_15_3 : 0);
    int cnt_w_15_4 = s_m[4388];
    n_kept += cnt_w_15_4;
    write_pos += ((warp_1 > 4) ? cnt_w_15_4 : 0);
    int cnt_w_15_5 = s_m[4389];
    n_kept += cnt_w_15_5;
    write_pos += ((warp_1 > 5) ? cnt_w_15_5 : 0);
    int cnt_w_15_6 = s_m[4390];
    n_kept += cnt_w_15_6;
    write_pos += ((warp_1 > 6) ? cnt_w_15_6 : 0);
    int cnt_w_15_7 = s_m[4391];
    n_kept += cnt_w_15_7;
    write_pos += ((warp_1 > 7) ? cnt_w_15_7 : 0);
    int cnt_w_15_8 = s_m[4392];
    n_kept += cnt_w_15_8;
    write_pos += ((warp_1 > 8) ? cnt_w_15_8 : 0);
    int cnt_w_15_9 = s_m[4393];
    n_kept += cnt_w_15_9;
    write_pos += ((warp_1 > 9) ? cnt_w_15_9 : 0);
    int cnt_w_15_10 = s_m[4394];
    n_kept += cnt_w_15_10;
    write_pos += ((warp_1 > 10) ? cnt_w_15_10 : 0);
    int cnt_w_15_11 = s_m[4395];
    n_kept += cnt_w_15_11;
    write_pos += ((warp_1 > 11) ? cnt_w_15_11 : 0);
    int cnt_w_15_12 = s_m[4396];
    n_kept += cnt_w_15_12;
    write_pos += ((warp_1 > 12) ? cnt_w_15_12 : 0);
    int cnt_w_15_13 = s_m[4397];
    n_kept += cnt_w_15_13;
    write_pos += ((warp_1 > 13) ? cnt_w_15_13 : 0);
    int cnt_w_15_14 = s_m[4398];
    n_kept += cnt_w_15_14;
    write_pos += ((warp_1 > 14) ? cnt_w_15_14 : 0);
    int cnt_w_15_15 = s_m[4399];
    n_kept += cnt_w_15_15;
    write_pos += ((warp_1 > 15) ? cnt_w_15_15 : 0);
    if (res == 0) {
        #pragma unroll 1
        for (int e0_3 = 0; e0_3 < n; e0_3 += 4096) {
            unsigned long long ent_w[8];
            #pragma unroll
            for (int u_6 = 0; u_6 < 8; u_6++) {
                int idx_l_1 = e0_3 + tid + 512 * u_6;
                ent_w[u_6] = 4294967295ll;
                if (idx_l_1 < n) {
                    ent_w[u_6] = (unsigned long long)Staging[base + (long long)idx_l_1];
                }
            }
            #pragma unroll
            for (int u_7 = 0; u_7 < 8; u_7++) {
                int kid_3 = (int)(unsigned int)(ent_w[u_7] & 4294967295ull);
                int kid_w = kid_3;
                int keep_w = 0;
                if (kid_w != -1) {
                    unsigned int bits_2 = (unsigned int)(ent_w[u_7] >> 32ull);
                    unsigned int magnitude_2 = bits_2 & 2147483647;
                    unsigned int m_2 = ((magnitude_2 != 0) ? bits_2 : zero_u);
                    unsigned int key32_2 = (((m_2 & sign_mask) != 0) ? ~m_2 : m_2 | sign_mask);
                    if (magnitude_2 > nan_floor) {
                        key32_2 = zero_u;
                    }
                    unsigned long long key64_1 = (unsigned long long)key32_2 << 32ull | (unsigned long long)(unsigned int)kid_w;
                    keep_w = ((key64_1 >= edge) ? 1 : 0);
                }
                unsigned int ballot_w = __ballot_sync(0xFFFFFFFF, keep_w != 0);
                if (keep_w != 0) {
                    int _rank_in_mask_4 = (int)__popc(ballot_w & ((1u << (unsigned int)lane) - 1u));
                    int p_w = write_pos + _rank_in_mask_4;
                    unsigned int bits_3 = (unsigned int)(ent_w[u_7] >> 32ull);
                    s_kid[p_w] = kid_w;
                    s_kb[p_w] = bits_3;
                    unsigned int x_k = (unsigned int)kid_w;
                    atomicAdd(&s_bm[(int)(x_k >> 5)], one_r << (x_k & 31));
                }
                write_pos += (int)(unsigned int)__popc(ballot_w);
            }
        }
    } else {
        unsigned int e_hi = (unsigned int)(edge >> 32ull);
        unsigned int e_lo = (unsigned int)(edge & 4294967295ull);
        #pragma unroll 1
        for (int e0_4 = 0; e0_4 < n_r; e0_4 += 8192) {
            unsigned long long ent_r[16];
            unsigned int kw[16];
            #pragma unroll
            for (int u_8 = 0; u_8 < 16; u_8++) {
                int idx_w = e0_4 + tid + 512 * u_8;
                kw[u_8] = s_k[idx_w];
                ent_r[u_8] = 4294967295ll;
                if (kw[u_8] != key_pad && kw[u_8] >= e_hi) {
                    ent_r[u_8] = (unsigned long long)Staging[base + (long long)idx_w];
                }
            }
            #pragma unroll
            for (int u_9 = 0; u_9 < 16; u_9++) {
                int kid_4 = (int)(unsigned int)(ent_r[u_9] & 4294967295ull);
                int kid_r = kid_4;
                int keep_r = 0;
                if (kid_r != -1) {
                    keep_r = ((kw[u_9] != e_hi || e_lo == 0 || (unsigned int)kid_r >= e_lo) ? 1 : 0);
                }
                unsigned int ballot_r = __ballot_sync(0xFFFFFFFF, keep_r != 0);
                if (keep_r != 0) {
                    int _rank_in_mask_5 = (int)__popc(ballot_r & ((1u << (unsigned int)lane) - 1u));
                    int p_r = write_pos + _rank_in_mask_5;
                    unsigned int bits_4 = (unsigned int)(ent_r[u_9] >> 32ull);
                    s_kid[p_r] = kid_r;
                    s_kb[p_r] = bits_4;
                    unsigned int x_k_1 = (unsigned int)kid_r;
                    atomicAdd(&s_bm[(int)(x_k_1 >> 5)], one_r << (x_k_1 & 31));
                }
                write_pos += (int)(unsigned int)__popc(ballot_r);
            }
        }
    }
    unsigned int pad_bits = 4286578688ll;
    float pad_score = 0.0f;
    pad_score = reinterpret_cast<float*>(&pad_bits)[0];
    __syncthreads();
    int w_a_0 = tid;
    unsigned int bw_a_0 = zero_u;
    if (w_a_0 < 256) {
        bw_a_0 = s_bm[w_a_0];
    }
    unsigned int c_a_0 = (unsigned int)__popc(bw_a_0);
    uint32_t incl_a_0 = c_a_0;
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_a_0, 1, 32); if (lane >= 1) { incl_a_0 += peer; } }
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_a_0, 2, 32); if (lane >= 2) { incl_a_0 += peer; } }
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_a_0, 4, 32); if (lane >= 4) { incl_a_0 += peer; } }
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_a_0, 8, 32); if (lane >= 8) { incl_a_0 += peer; } }
    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, incl_a_0, 16, 32); if (lane >= 16) { incl_a_0 += peer; } }
    if (w_a_0 < 256) {
        s_pref[w_a_0] = (uint16_t)(incl_a_0 - c_a_0);
    }
    if (lane_1 == 31) {
        s_xr[warp_1] = incl_a_0;
    }
    __syncthreads();
    unsigned int running = zero_u;
    int w_b_0 = tid;
    unsigned int mine_0 = running;
    unsigned int slab_tot_0 = zero_u;
    unsigned int tot_b_0 = s_xr[0];
    mine_0 += ((warp_1 > 0) ? tot_b_0 : zero_u);
    slab_tot_0 += tot_b_0;
    unsigned int tot_b_1 = s_xr[1];
    mine_0 += ((warp_1 > 1) ? tot_b_1 : zero_u);
    slab_tot_0 += tot_b_1;
    unsigned int tot_b_2 = s_xr[2];
    mine_0 += ((warp_1 > 2) ? tot_b_2 : zero_u);
    slab_tot_0 += tot_b_2;
    unsigned int tot_b_3 = s_xr[3];
    mine_0 += ((warp_1 > 3) ? tot_b_3 : zero_u);
    slab_tot_0 += tot_b_3;
    unsigned int tot_b_4 = s_xr[4];
    mine_0 += ((warp_1 > 4) ? tot_b_4 : zero_u);
    slab_tot_0 += tot_b_4;
    unsigned int tot_b_5 = s_xr[5];
    mine_0 += ((warp_1 > 5) ? tot_b_5 : zero_u);
    slab_tot_0 += tot_b_5;
    unsigned int tot_b_6 = s_xr[6];
    mine_0 += ((warp_1 > 6) ? tot_b_6 : zero_u);
    slab_tot_0 += tot_b_6;
    unsigned int tot_b_7 = s_xr[7];
    mine_0 += ((warp_1 > 7) ? tot_b_7 : zero_u);
    slab_tot_0 += tot_b_7;
    unsigned int tot_b_8 = s_xr[8];
    mine_0 += ((warp_1 > 8) ? tot_b_8 : zero_u);
    slab_tot_0 += tot_b_8;
    unsigned int tot_b_9 = s_xr[9];
    mine_0 += ((warp_1 > 9) ? tot_b_9 : zero_u);
    slab_tot_0 += tot_b_9;
    unsigned int tot_b_10 = s_xr[10];
    mine_0 += ((warp_1 > 10) ? tot_b_10 : zero_u);
    slab_tot_0 += tot_b_10;
    unsigned int tot_b_11 = s_xr[11];
    mine_0 += ((warp_1 > 11) ? tot_b_11 : zero_u);
    slab_tot_0 += tot_b_11;
    unsigned int tot_b_12 = s_xr[12];
    mine_0 += ((warp_1 > 12) ? tot_b_12 : zero_u);
    slab_tot_0 += tot_b_12;
    unsigned int tot_b_13 = s_xr[13];
    mine_0 += ((warp_1 > 13) ? tot_b_13 : zero_u);
    slab_tot_0 += tot_b_13;
    unsigned int tot_b_14 = s_xr[14];
    mine_0 += ((warp_1 > 14) ? tot_b_14 : zero_u);
    slab_tot_0 += tot_b_14;
    unsigned int tot_b_15 = s_xr[15];
    mine_0 += ((warp_1 > 15) ? tot_b_15 : zero_u);
    slab_tot_0 += tot_b_15;
    if (w_b_0 < 256) {
        unsigned int lane_b_0 = (unsigned int)s_pref[w_b_0];
        s_pref[w_b_0] = (uint16_t)(mine_0 + lane_b_0);
    }
    running += slab_tot_0;
    __syncthreads();
    int p_o_0 = tid;
    if (p_o_0 < n_kept) {
        int kid_o_0 = s_kid[p_o_0];
        unsigned int bits_o_0 = s_kb[p_o_0];
        unsigned int x_o_0 = (unsigned int)kid_o_0;
        int w_o_0 = (int)(x_o_0 >> 5);
        unsigned int below_o_0 = (one_r << (x_o_0 & 31)) - one_r;
        unsigned int bw_o_0 = s_bm[w_o_0];
        unsigned int pref_o_0 = (unsigned int)s_pref[w_o_0];
        int dst_o_0 = (int)(pref_o_0 + (unsigned int)__popc(bw_o_0 & below_o_0));
        s_oid[dst_o_0] = kid_o_0;
        s_osc[dst_o_0] = bits_o_0;
    }
    int p_o_1 = 512 + tid;
    if (p_o_1 < n_kept) {
        int kid_o_1 = s_kid[p_o_1];
        unsigned int bits_o_1 = s_kb[p_o_1];
        unsigned int x_o_1 = (unsigned int)kid_o_1;
        int w_o_1 = (int)(x_o_1 >> 5);
        unsigned int below_o_1 = (one_r << (x_o_1 & 31)) - one_r;
        unsigned int bw_o_1 = s_bm[w_o_1];
        unsigned int pref_o_1 = (unsigned int)s_pref[w_o_1];
        int dst_o_1 = (int)(pref_o_1 + (unsigned int)__popc(bw_o_1 & below_o_1));
        s_oid[dst_o_1] = kid_o_1;
        s_osc[dst_o_1] = bits_o_1;
    }
    int p_o_2 = 1024 + tid;
    if (p_o_2 < n_kept) {
        int kid_o_2 = s_kid[p_o_2];
        unsigned int bits_o_2 = s_kb[p_o_2];
        unsigned int x_o_2 = (unsigned int)kid_o_2;
        int w_o_2 = (int)(x_o_2 >> 5);
        unsigned int below_o_2 = (one_r << (x_o_2 & 31)) - one_r;
        unsigned int bw_o_2 = s_bm[w_o_2];
        unsigned int pref_o_2 = (unsigned int)s_pref[w_o_2];
        int dst_o_2 = (int)(pref_o_2 + (unsigned int)__popc(bw_o_2 & below_o_2));
        s_oid[dst_o_2] = kid_o_2;
        s_osc[dst_o_2] = bits_o_2;
    }
    int p_o_3 = 1536 + tid;
    if (p_o_3 < n_kept) {
        int kid_o_3 = s_kid[p_o_3];
        unsigned int bits_o_3 = s_kb[p_o_3];
        unsigned int x_o_3 = (unsigned int)kid_o_3;
        int w_o_3 = (int)(x_o_3 >> 5);
        unsigned int below_o_3 = (one_r << (x_o_3 & 31)) - one_r;
        unsigned int bw_o_3 = s_bm[w_o_3];
        unsigned int pref_o_3 = (unsigned int)s_pref[w_o_3];
        int dst_o_3 = (int)(pref_o_3 + (unsigned int)__popc(bw_o_3 & below_o_3));
        s_oid[dst_o_3] = kid_o_3;
        s_osc[dst_o_3] = bits_o_3;
    }
    __syncthreads();
    int p_s_0 = tid;
    if (p_s_0 < top_k) {
        long long slot_s_0 = row_base + (long long)p_s_0;
        if (p_s_0 < n_kept) {
            float sc_s_0 = 0.0f;
            unsigned int bits_s_0 = s_osc[p_s_0];
            sc_s_0 = reinterpret_cast<float*>(&bits_s_0)[0];
            int id_s_0 = s_oid[p_s_0];
            Indices[slot_s_0] = id_s_0;
            Scores[slot_s_0] = sc_s_0;
        } else {
            Indices[slot_s_0] = -1;
            Scores[slot_s_0] = pad_score;
        }
    }
    int p_s_1 = 512 + tid;
    if (p_s_1 < top_k) {
        long long slot_s_1 = row_base + (long long)p_s_1;
        if (p_s_1 < n_kept) {
            float sc_s_1 = 0.0f;
            unsigned int bits_s_1 = s_osc[p_s_1];
            sc_s_1 = reinterpret_cast<float*>(&bits_s_1)[0];
            int id_s_1 = s_oid[p_s_1];
            Indices[slot_s_1] = id_s_1;
            Scores[slot_s_1] = sc_s_1;
        } else {
            Indices[slot_s_1] = -1;
            Scores[slot_s_1] = pad_score;
        }
    }
    int p_s_2 = 1024 + tid;
    if (p_s_2 < top_k) {
        long long slot_s_2 = row_base + (long long)p_s_2;
        if (p_s_2 < n_kept) {
            float sc_s_2 = 0.0f;
            unsigned int bits_s_2 = s_osc[p_s_2];
            sc_s_2 = reinterpret_cast<float*>(&bits_s_2)[0];
            int id_s_2 = s_oid[p_s_2];
            Indices[slot_s_2] = id_s_2;
            Scores[slot_s_2] = sc_s_2;
        } else {
            Indices[slot_s_2] = -1;
            Scores[slot_s_2] = pad_score;
        }
    }
    int p_s_3 = 1536 + tid;
    if (p_s_3 < top_k) {
        long long slot_s_3 = row_base + (long long)p_s_3;
        if (p_s_3 < n_kept) {
            float sc_s_3 = 0.0f;
            unsigned int bits_s_3 = s_osc[p_s_3];
            sc_s_3 = reinterpret_cast<float*>(&bits_s_3)[0];
            int id_s_3 = s_oid[p_s_3];
            Indices[slot_s_3] = id_s_3;
            Scores[slot_s_3] = sc_s_3;
        } else {
            Indices[slot_s_3] = -1;
            Scores[slot_s_3] = pad_score;
        }
    }
}

} // extern "C"
