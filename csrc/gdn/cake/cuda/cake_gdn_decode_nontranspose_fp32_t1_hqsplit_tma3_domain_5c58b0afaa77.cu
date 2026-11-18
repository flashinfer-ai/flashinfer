// Copyright (c) 2026 by FlashInfer team.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// clang-format off
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

template <typename T, int N>
struct CakeParamArray {
    T v[N];
    __device__ __forceinline__ const T& operator[](int i) const { return v[i]; }
};

#ifndef H
#error "H is a downstream specialization of this program; define it on the compile line"
#elif H != 16
#error "H takes 16 in this program; another value renders other code"
#endif
#ifndef HV
#error "HV is a downstream specialization of this program; define it on the compile line"
#elif HV != 32
#error "HV takes 32 in this program; another value renders other code"
#endif
#ifndef SCALE
#error "SCALE is a downstream specialization of this program; define it on the compile line"
#endif

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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}





__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}


// Reference form of the product below: the same expression the generated
// source spells on other targets, kept out of line so its FP64 instructions sit
// behind a call the normal-input path never takes (an if-converted block would
// still issue them).  Every rounding and flush-to-zero policy of the compile line
// applies to it unchanged, so the callers below are exact for every input.
__device__ __noinline__ float cake_f32_mul_f64const_rn_reference(float f64c_x, double f64c_c) {
    return (float)((double)f64c_x * f64c_c);
}

// Bitwise (float)((double)x * c) without FP64 arithmetic on the common path.
//
// The double multiply rounds the exact 76/77-bit significand product to 53 bits
// (round to nearest even), the float conversion rounds that to 24 bits (round to
// nearest even again: the result is NOT always the correctly rounded f32 of the
// exact product).  Both roundings and both carry-outs are emulated with integer
// arithmetic.  `c` is a literal at every call site (a kernel #define), so its
// sign, exponent and significand halves fold to immediates: the hot path is two
// 32x32 -> 64-bit multiplies, shifts, masks and compares.
//
// Exactness argument (x normal, c normal, result a normal f32):
//   P = Mx * Mc with Mx in [2^23, 2^24), Mc in [2^52, 2^53)  =>  P in [2^75, 2^77).
//   The double result D is P rounded to 53 significant bits; D's magnitude lies
//   within one rounding of the f32 result, so D is a normal double whenever the
//   f32 result is normal (|D| in [2^-128, 2^129) is far inside the double range)
//   and the 53-bit rounding position is exactly the one the hardware uses.
//   The float result is D rounded to 24 significant bits; a result whose final
//   biased exponent leaves [1, 254] (subnormal, zero after rounding, overflow)
//   is not produced here but by the reference form.  A carry out of either
//   rounding (2^53 -> 2^52, 2^24 -> 2^23) increments the exponent before the
//   range check, so a value that rounds up into the normal range is handled by
//   the integer path only when the hardware also rounds it to that normal.
// Fast path, folded at compile time: c exactly representable in f32 (every power
// of two of the f32 range, 1.0, 0.125, 0.0625, 2^-15, +-0, +-inf, ...).  The
// double product of two f32 values is exact (24 + 24 < 53 significant bits), so
// its one f32 rounding equals the double-then-float rounding for every finite x
// with a normal result; and the compiler already folds the incumbent text
// `(float)((double)x * C)` to this very f32 multiply when C shrinks to f32 exactly
// (fptrunc(fmul(fpext x, C)) -> fmul x, (float)C), so NaN payloads and the compile
// line's flush-to-zero treatment of subnormal inputs/results follow the f32
// multiply exactly as the folded incumbent does (1.0 folds further to x itself).
// Cases routed to the reference form: x subnormal/inf/NaN, c subnormal/NaN
// (compile-time folded), a result outside the normal range.
__device__ __forceinline__ float cake_f32_mul_f64const_rn(float f64c_x, double f64c_c) {
    if ((double)(float)f64c_c == f64c_c) {
        return f64c_x * (float)f64c_c;
    }
    const unsigned long long f64c_cb = (unsigned long long)__double_as_longlong(f64c_c);
    const unsigned int f64c_ce = (unsigned int)((f64c_cb >> 52) & 0x7FFull);
    const unsigned int f64c_xb = __float_as_uint(f64c_x);
    const unsigned int f64c_xe = (f64c_xb >> 23) & 0xFFu;
    const unsigned int f64c_sign = (f64c_xb ^ (unsigned int)(f64c_cb >> 32)) & 0x80000000u;
    if (f64c_ce == 0u || f64c_ce == 0x7FFu) {
        return cake_f32_mul_f64const_rn_reference(f64c_x, f64c_c);  // subnormal or NaN c
    }
    if (f64c_xe == 0u) {
        if ((f64c_xb & 0x7FFFFFu) == 0u) {
            return __uint_as_float(f64c_sign);  // (+-0) * finite nonzero c = +-0 (sign xor), exactly
        }
        return cake_f32_mul_f64const_rn_reference(f64c_x, f64c_c);  // subnormal x
    }
    if (__builtin_expect(f64c_xe == 0xFFu, 0)) {
        return cake_f32_mul_f64const_rn_reference(f64c_x, f64c_c);
    }
    const unsigned int f64c_xm = (f64c_xb & 0x7FFFFFu) | 0x800000u;                   // 24-bit significand of x
    const unsigned long long f64c_cm = (f64c_cb & 0xFFFFFFFFFFFFFull) | (1ull << 52);  // 53-bit significand of c
    const unsigned int f64c_cm_lo = (unsigned int)(f64c_cm & 0xFFFFFFFFull);
    const unsigned int f64c_cm_hi = (unsigned int)(f64c_cm >> 32);
    // Exact product P = xm * cm (< 2^77) as (P >> 24, P & 0xFFFFFF).
    const unsigned long long f64c_p_lo = (unsigned long long)f64c_xm * f64c_cm_lo;
    const unsigned long long f64c_p_hi = (unsigned long long)f64c_xm * f64c_cm_hi;
    const unsigned long long f64c_q = (f64c_p_hi << 8) + (f64c_p_lo >> 24);  // P >> 24, in [2^51, 2^53)
    const unsigned int f64c_low = (unsigned int)(f64c_p_lo & 0xFFFFFFu);      // P & 0xFFFFFF
    // Biased f32 exponent of the result when P < 2^76; one more when P >= 2^76.
    int f64c_exp = (int)f64c_xe + (int)f64c_ce - 1023;
    unsigned long long f64c_m53;
    unsigned int f64c_round;
    unsigned int f64c_sticky;
    if (f64c_q >> 52) {
        // 77 significant bits: m53 = P >> 24, round bit 23, sticky bits 0..22.
        f64c_m53 = f64c_q;
        f64c_round = (f64c_low >> 23) & 1u;
        f64c_sticky = (f64c_low & 0x7FFFFFu) != 0u;
        f64c_exp += 1;
    } else {
        // 76 significant bits: m53 = P >> 23, round bit 22, sticky bits 0..21.
        f64c_m53 = (f64c_q << 1) | (unsigned long long)((f64c_low >> 23) & 1u);
        f64c_round = (f64c_low >> 22) & 1u;
        f64c_sticky = (f64c_low & 0x3FFFFFu) != 0u;
    }
    // Rounding 1 (the double multiply): exact product -> 53 bits, nearest even.
    f64c_m53 += (unsigned long long)(f64c_round & (f64c_sticky | (unsigned int)(f64c_m53 & 1ull)));
    if (f64c_m53 >> 53) {
        f64c_m53 >>= 1;  // 2^53 -> 2^52
        f64c_exp += 1;
    }
    // Rounding 2 (the float conversion): 53 bits -> 24 bits, nearest even.
    unsigned int f64c_m24 = (unsigned int)(f64c_m53 >> 29);
    const unsigned int f64c_round2 = (unsigned int)(f64c_m53 >> 28) & 1u;
    const unsigned int f64c_sticky2 = (f64c_m53 & 0xFFFFFFFull) != 0ull;
    f64c_m24 += f64c_round2 & (f64c_sticky2 | (f64c_m24 & 1u));
    if (f64c_m24 >> 24) {
        f64c_m24 >>= 1;  // 2^24 -> 2^23
        f64c_exp += 1;
    }
    if (__builtin_expect((unsigned int)(f64c_exp - 1) >= 254u, 0)) {
        return cake_f32_mul_f64const_rn_reference(f64c_x, f64c_c);  // not a normal f32 result
    }
    return __uint_as_float(f64c_sign | ((unsigned int)f64c_exp << 23) | (f64c_m24 & 0x7FFFFFu));
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_gdn_decode_nontranspose_fp32_t1_hqsplit_tma3_domain(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, const __grid_constant__ CUtensorMap state_tma, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, long long q_stride_p0, long long k_stride_p0, long long v_stride_p0, long long a_stride_p0, long long b_stride_p0, long long q_size_p0, int rows0, int* __restrict__ counters0, int* __restrict__ counters1, const CakeParamArray<int, 256> sm_table)
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

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* sState = reinterpret_cast<float*>(smem_raw + 1024);
    const int sState_addr = smem + 1024;
    float* sQ = reinterpret_cast<float*>(smem_raw + 50176);
    const int sQ_addr = smem + 50176;
    float* sK = reinterpret_cast<float*>(smem_raw + 50688);
    const int sK_addr = smem + 50688;
    int* sword = reinterpret_cast<int*>(smem_raw + 51232);
    const int sword_addr = smem + 51232;

    // Mbarrier init (3 barriers at smem_raw[0..24))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'panel_pipe' ---
            // full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    int rows = (int)q_size_p0;
    int items0 = rows0 * HV;
    int items1 = (rows - rows0) * HV;
    int ctas = (int)gridDim.x;
    int lane_local = lane;
    int warp_local = warp;
    int k_local = lane_local / 4;
    int v_local = lane_local - k_local * 4;
    int tile_v_local = warp_local * 4 + v_local;
    int swz_col = (warp_local ^ k_local) * 4 + v_local;
    float r_q[4];
    float r_k[4];
    float r_h[16];
    int home = 0;
    int visit = 0;
    if (tid == 224) {
        uint32_t _read_special_reg_0;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(_read_special_reg_0));
        int sm = (int)_read_special_reg_0;
        int entry = sm_table[sm];
        uint32_t _read_special_reg_1;
        asm("mov.u32 %0, %%locality_domain_id;" : "=r"(_read_special_reg_1));
        int hw_die = (int)_read_special_reg_1;
        int table_die = entry >> 20 & 15;
        int die_xor = table_die ^ hw_die;
        home = table_die;
        int v_0 = visit;
        int word = -1;
        if (v_0 == 0) {
            int got = 0;
            int tiles = items0;
            if (home == 0) {
                int _atomic_fetch_add_2 = atomicAdd(counters0, 1);
                got = _atomic_fetch_add_2;
                if (got == items0 + ctas - 1) {
                    counters0[0] = 0;
                }
            } else {
                tiles = items1;
                int _atomic_fetch_add_3 = atomicAdd(counters1, 1);
                got = _atomic_fetch_add_3;
                if (got == items1 + ctas - 1) {
                    counters1[0] = 0;
                }
            }
            if (got < tiles) {
                word = got * 2 + home;
            } else {
                v_0 = 1;
            }
        }
        if (v_0 == 1) {
            int other = home ^ 1;
            int got_1 = 0;
            int tiles_1 = items0;
            if (other == 0) {
                int _atomic_fetch_add_4 = atomicAdd(counters0, 1);
                got_1 = _atomic_fetch_add_4;
                if (got_1 == items0 + ctas - 1) {
                    counters0[0] = 0;
                }
            } else {
                tiles_1 = items1;
                int _atomic_fetch_add_5 = atomicAdd(counters1, 1);
                got_1 = _atomic_fetch_add_5;
                if (got_1 == items1 + ctas - 1) {
                    counters1[0] = 0;
                }
            }
            if (got_1 < tiles_1) {
                word = got_1 * 2 + other;
            } else {
                v_0 = 2;
            }
        }
        visit = v_0;
        sword[0] = word;
    }
    __syncthreads();
    int cur;
    asm volatile("redux.sync.min.s32 %0, %1, 0xFFFFFFFF;" : "=r"(cur) : "r"(sword[0]));
    if (cur != -1) {
        int item = cur >> 1;
        int row = item / HV;
        int head = item - row * HV;
        if ((cur & 1) == 1) {
            row = row + rows0;
        }
        if (warp == 0) {
            if (elect_sync()) {
                #pragma unroll
                for (int panel = 0; panel < 2; panel++) {
                    mbarrier_arrive_expect_tx(full_addr + (panel) * 8, 16384);
                    tma_4d_gmem2smem(sState_addr + (unsigned int)(panel * 16384), (&state_tma), panel * 32, 0, head, row, full_addr + (panel) * 8);
                }
            }
        }
    }
    int cs = 0;
    int trip = 1;
    unsigned int _phase_full = 0;
    while (cur != -1) {
        int item_1 = cur >> 1;
        int row_1 = item_1 / HV;
        int head_1 = item_1 - row_1 * HV;
        if ((cur & 1) == 1) {
            row_1 = row_1 + rows0;
        }
        if (tid == 224) {
            int v_1 = visit;
            int word_1 = -1;
            if (v_1 == 0) {
                int got_2 = 0;
                int tiles_2 = items0;
                if (home == 0) {
                    int _atomic_fetch_add_6 = atomicAdd(counters0, 1);
                    got_2 = _atomic_fetch_add_6;
                    if (got_2 == items0 + ctas - 1) {
                        counters0[0] = 0;
                    }
                } else {
                    tiles_2 = items1;
                    int _atomic_fetch_add_7 = atomicAdd(counters1, 1);
                    got_2 = _atomic_fetch_add_7;
                    if (got_2 == items1 + ctas - 1) {
                        counters1[0] = 0;
                    }
                }
                if (got_2 < tiles_2) {
                    word_1 = got_2 * 2 + home;
                } else {
                    v_1 = 1;
                }
            }
            if (v_1 == 1) {
                int other_1 = home ^ 1;
                int got_3 = 0;
                int tiles_3 = items0;
                if (other_1 == 0) {
                    int _atomic_fetch_add_8 = atomicAdd(counters0, 1);
                    got_3 = _atomic_fetch_add_8;
                    if (got_3 == items0 + ctas - 1) {
                        counters0[0] = 0;
                    }
                } else {
                    tiles_3 = items1;
                    int _atomic_fetch_add_9 = atomicAdd(counters1, 1);
                    got_3 = _atomic_fetch_add_9;
                    if (got_3 == items1 + ctas - 1) {
                        counters1[0] = 0;
                    }
                }
                if (got_3 < tiles_3) {
                    word_1 = got_3 * 2 + other_1;
                } else {
                    v_1 = 2;
                }
            }
            visit = v_1;
            sword[trip & 1] = word_1;
        }
        int qk_h = head_1 / (HV / H);
        long long q_base = (long long)row_1 * q_stride_p0 + (long long)qk_h * 128ll;
        long long k_base = (long long)row_1 * k_stride_p0 + (long long)qk_h * 128ll;
        long long v_base = (long long)row_1 * v_stride_p0 + (long long)head_1 * 128ll;
        int out_base = (row_1 * HV + head_1) * 128;
        if (warp_local == 0) {
            int qk_lane_base = lane_local * 4;
            {
                uint2 _vld_10;
                _vld_10 = *reinterpret_cast<const uint2*>(q + q_base + (long long)qk_lane_base);
                uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&r_q[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                    (&r_q[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_11;
                _vld_11 = *reinterpret_cast<const uint2*>(k + k_base + (long long)qk_lane_base);
                uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&r_k[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
                    (&r_k[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
                }
            }
            float sum_q = 0.0f;
            float sum_k = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                sum_q = __fmaf_rn(r_q[i], r_q[i], sum_q);
                sum_k = __fmaf_rn(r_k[i], r_k[i], sum_k);
            }
            float _warp_reduce_12 = sum_q;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_12 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_12, offset);
            sum_q = _warp_reduce_12;
            float _warp_reduce_13 = sum_k;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_13 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_13, offset);
            sum_k = _warp_reduce_13;
            float _rsqrt_14 = rsqrtf(sum_q + 1e-06f);
            float q_norm = cake_f32_mul_f64const_rn(_rsqrt_14, SCALE);
            float k_norm = rsqrtf(sum_k + 1e-06f);
            #pragma unroll
            for (int i_1 = 0; i_1 < 4; i_1++) {
                sQ[qk_lane_base + i_1] = r_q[i_1] * q_norm;
                sK[qk_lane_base + i_1] = r_k[i_1] * k_norm;
            }
        }
        float decay_lane = 0.0f;
        float beta_lane = 0.0f;
        if (lane_local == 0) {
            float gate_x = (float)a[(long long)row_1 * a_stride_p0 + (long long)head_1] + dt_bias[head_1];
            float softplus_x = gate_x;
            if (gate_x <= 20.0f) {
                float _exp_16 = __expf(gate_x);
                float _log2_15;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_15) : "f"(1.0f + _exp_16));
                softplus_x = _log2_15 * 0.6931472f;
            }
            float _exp_18 = __expf(A_log[head_1]);
            float _exp_17 = __expf(-_exp_18 * softplus_x);
            decay_lane = _exp_17;
            float _exp_20 = __expf(-(float)b[(long long)row_1 * b_stride_p0 + (long long)head_1]);
            float _rcp_19 = approx_rcp(1.0f + _exp_20);
            beta_lane = _rcp_19;
        }
        float decay_val = __shfl_sync(0xFFFFFFFF, decay_lane, 0);
        float beta_val = __shfl_sync(0xFFFFFFFF, beta_lane, 0);
        __syncthreads();
        int nxt;
        asm volatile("redux.sync.min.s32 %0, %1, 0xFFFFFFFF;" : "=r"(nxt) : "r"(sword[trip & 1]));
        int item_0 = nxt >> 1;
        int row_0 = item_0 / HV;
        int head_0 = item_0 - row_0 * HV;
        if ((nxt & 1) == 1) {
            row_0 = row_0 + rows0;
        }
        #pragma unroll
        for (int tile = 0; tile < 4; tile++) {
            unsigned int stage_base = sState_addr + (unsigned int)(cs * 16384);
            int stage_elems = cs * 4096;
            int tile_v_base = tile * 32;
            if (tile < 2) {
                if (warp == 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive_expect_tx(full_addr + (((cs + 2 >= 3) ? cs + 2 - 3 : cs + 2)) * 8, 16384);
                        tma_4d_gmem2smem(sState_addr + (unsigned int)(((cs + 2 >= 3) ? cs + 2 - 3 : cs + 2) * 16384), (&state_tma), (tile + 2) * 32, 0, head_1, row_1, full_addr + (((cs + 2 >= 3) ? cs + 2 - 3 : cs + 2)) * 8);
                    }
                }
            } else if (nxt != -1) {
                if (warp == 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive_expect_tx(full_addr + (((cs + 2 >= 3) ? cs + 2 - 3 : cs + 2)) * 8, 16384);
                        tma_4d_gmem2smem(sState_addr + (unsigned int)(((cs + 2 >= 3) ? cs + 2 - 3 : cs + 2) * 16384), (&state_tma), (tile - 2) * 32, 0, head_0, row_0, full_addr + (((cs + 2 >= 3) ? cs + 2 - 3 : cs + 2)) * 8);
                    }
                }
            }
            mbarrier_wait(full_addr + (cs) * 8, _phase_full);
            float sum_hk = 0.0f;
            #pragma unroll
            for (int k_iter = 0; k_iter < 16; k_iter++) {
                int k_idx = k_iter * 8 + k_local;
                float h_val = sState[stage_elems + k_idx * 32 + swz_col] * decay_val;
                r_h[k_iter] = h_val;
                sum_hk = __fmaf_rn(h_val, sK[k_idx], sum_hk);
            }
            float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, sum_hk, 16);
            sum_hk += _shfl_xor_sync_21;
            float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, sum_hk, 8);
            sum_hk += _shfl_xor_sync_22;
            float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, sum_hk, 4);
            sum_hk += _shfl_xor_sync_23;
            float value_lane = 0.0f;
            if (k_local == 0) {
                value_lane = (float)v[v_base + (long long)tile_v_base + (long long)tile_v_local];
            }
            float value_val = __shfl_sync(0xFFFFFFFF, value_lane, v_local);
            float delta = (value_val - sum_hk) * beta_val;
            float sum_hq = 0.0f;
            #pragma unroll
            for (int k_iter_1 = 0; k_iter_1 < 16; k_iter_1++) {
                int k_idx_1 = k_iter_1 * 8 + k_local;
                float h_new = __fmaf_rn(sK[k_idx_1], delta, r_h[k_iter_1]);
                r_h[k_iter_1] = h_new;
                sum_hq = __fmaf_rn(h_new, sQ[k_idx_1], sum_hq);
            }
            #pragma unroll
            for (int k_iter_2 = 0; k_iter_2 < 16; k_iter_2++) {
                int k_idx_2 = k_iter_2 * 8 + k_local;
                sState[stage_elems + k_idx_2 * 32 + swz_col] = r_h[k_iter_2];
            }
            float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, sum_hq, 16);
            sum_hq += _shfl_xor_sync_24;
            float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, sum_hq, 8);
            sum_hq += _shfl_xor_sync_25;
            float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, sum_hq, 4);
            sum_hq += _shfl_xor_sync_26;
            if (k_local == 0) {
                out[out_base + tile_v_base + tile_v_local] = sum_hq;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (warp == 0) {
                if (elect_sync()) {
                    tma_store_4d((&state_tma), tile * 32, 0, head_1, row_1, stage_base);
                    asm volatile("cp.async.bulk.commit_group;");
                }
            }
            cs += 1;
            if (cs == 3) { cs = 0; _phase_full ^= 1; }
        }
        cur = nxt;
        trip = trip + 1;
    }
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("cp.async.bulk.wait_group 0;");
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
// clang-format on
