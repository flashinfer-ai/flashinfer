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

#define LAUNCH_MIN_BLOCKS 2

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




__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}





__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_affine_split_4cd76cf613d87c68257c(float* __restrict__ o_loc, __nv_bfloat16* __restrict__ c_rows, const __grid_constant__ CUtensorMap c_tma, float* __restrict__ bm, __nv_bfloat16* __restrict__ hpub, const __grid_constant__ CUtensorMap hpub_tma, __nv_bfloat16* __restrict__ output, float* __restrict__ output_state, unsigned int* __restrict__ flags, unsigned int* __restrict__ ticket, int* __restrict__ cu_seqlens, long long output_state_stride_slot, int total_tokens, int num_seqs, int num_heads, int p_cap, int c_min)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(hpub));
    __builtin_assume(__isGlobal(hpub));

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
    #define m_full_addr (mbar_base + 0)
    #define in_full_addr (mbar_base + 32)
    #define h_full_addr (mbar_base + 48)
    #define mma_done_addr (mbar_base + 56)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* ring_buf = reinterpret_cast<float*>(smem_raw + 1024);
    const int ring_addr = smem + 1024;
    float* xbuf = reinterpret_cast<float*>(smem_raw + 66560);
    const int x_addr = smem + 66560;
    float* part = reinterpret_cast<float*>(smem_raw + 78848);
    const int part_addr = smem + 78848;
    __nv_bfloat16* h_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int h_hi_addr = smem + 1024;
    __nv_bfloat16* h_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int h_lo_addr = smem + 17408;
    __nv_bfloat16* c_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int c_addr = smem + 33792;
    __nv_bfloat16* c_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int _c_1_addr = smem + 66560;
    unsigned int* role_cell = reinterpret_cast<unsigned int*>(smem_raw + 111616);
    const int role_cell_addr = smem + 111616;

    // Mbarrier init (8 barriers at smem_raw[0..64))

    if (warp == 0) {
        // m_full: 4 barriers, init_count=1
        // in_full: 2 barriers, init_count=1
        // h_full: 1 barriers, init_count=1
        // mma_done: 1 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        if (lane < 8) {
            mbarrier_init(smem + 0 + lane * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 64);
    if (warp == 0) {
        int _tmem_hold = smem + 64;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc = taddr;

    // === Task calls (declaration order) ===
    if (warp == 0) {
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    if (tid == 0) {
        uint32_t claimed;
        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(claimed) : "l"(ticket), "r"(static_cast<uint32_t>((unsigned int)(gridDim.x - 1))) : "memory");
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(role_cell_addr), "r"(claimed) : "memory");
    }
    __syncthreads();
    unsigned int cell[1];
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&cell[0])) : "r"(role_cell_addr));
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int role = (int)cell[0];
    int n_compose = num_seqs * num_heads * 16;
    int cuv[2];
    int join = 1;
    if (role < n_compose) {
        int n = role / (num_heads * 16);
        int h = role / 16 % num_heads;
        int v0 = role % 16 * 8;
        {
            uint32_t _scalar_bits_0;
            asm volatile("ld.global.nc.b32 %0, [%1];"
                : "=r"(_scalar_bits_0) : "l"((const void*)(cu_seqlens + (n))) : "memory");
            cuv[0] = (int32_t)_scalar_bits_0;
        }
        {
            uint32_t _scalar_bits_1;
            asm volatile("ld.global.nc.b32 %0, [%1];"
                : "=r"(_scalar_bits_1) : "l"((const void*)(cu_seqlens + (n + 1))) : "memory");
            cuv[1] = (int32_t)_scalar_bits_1;
        }
        int _max_2 = ((cuv[0]) > (0) ? (cuv[0]) : (0));
        int start = ((_max_2) < (total_tokens) ? (_max_2) : (total_tokens));
        int _max_5 = ((cuv[1]) > (start) ? (cuv[1]) : (start));
        int _min_4 = ((_max_5) < (total_tokens) ? (_max_5) : (total_tokens));
        int _min_3 = (((_min_4 - start + 63) / 64 / c_min) < (p_cap) ? ((_min_4 - start + 63) / 64 / c_min) : (p_cap));
        int segs = ((_min_3) > (1) ? (_min_3) : (1));
        if (segs >= 2) {
            int vg = 0;
            int rq = warp;
            int nchunks = (segs - 1) * 4;
            float b0[4];
            float bpre[4];
            int p_0 = tid;
            {
                float2 _v2_6 = *reinterpret_cast<const float2*>(bm + (((long long)(n * 2 * p_cap) * (long long)num_heads + (long long)h) * 16384ll + (long long)((v0 + p_0 / 64) * 128 + p_0 % 64 * 2)) + 0);
                b0[0] = _v2_6.x;
                b0[0 + 1] = _v2_6.y;
            }
            int p_1 = tid + 256;
            {
                float2 _v2_7 = *reinterpret_cast<const float2*>(bm + (((long long)(n * 2 * p_cap) * (long long)num_heads + (long long)h) * 16384ll + (long long)((v0 + p_1 / 64) * 128 + p_1 % 64 * 2)) + 0);
                b0[2] = _v2_7.x;
                b0[2 + 1] = _v2_7.y;
            }
            int p_2 = tid;
            {
                float2 _v2_8 = *reinterpret_cast<const float2*>(bm + (((long long)(n * 2 * p_cap + 1) * (long long)num_heads + (long long)h) * 16384ll + (long long)((v0 + p_2 / 64) * 128 + p_2 % 64 * 2)) + 0);
                bpre[0] = _v2_8.x;
                bpre[0 + 1] = _v2_8.y;
            }
            int p_3 = tid + 256;
            {
                float2 _v2_9 = *reinterpret_cast<const float2*>(bm + (((long long)(n * 2 * p_cap + 1) * (long long)num_heads + (long long)h) * 16384ll + (long long)((v0 + p_3 / 64) * 128 + p_3 % 64 * 2)) + 0);
                bpre[2] = _v2_9.x;
                bpre[2 + 1] = _v2_9.y;
            }
            if (tid == 0) {
                #pragma unroll 1
                for (int c = 0; c < 4; c++) {
                    if (c < nchunks) {
                        long long src = ((long long)((n * 2 + 1) * p_cap + (1 + c / 4)) * (long long)num_heads + (long long)h) * 16384ll + (long long)(c % 4 * 32 * 128);
                        mbarrier_arrive_expect_tx(m_full_addr + (c) * 8, 16384);
                        cp_async_bulk_gmem2smem(ring_addr + (unsigned int)(c * 16384), bm + src, 16384, m_full_addr + (c) * 8);
                    }
                }
            }
            for (int line = role % 16 * 256 + tid; line < (segs - 2) * 512; line += 4096) {
                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(bm + (((long long)((n * 2 + 1) * p_cap + (2 + line / 512)) * (long long)num_heads + (long long)h) * 16384ll + (long long)(line % 512 * 32)))));
            }
            int p_4 = tid;
            int v_0 = p_4 / 64;
            int k_0 = p_4 % 64 * 2;
            float b2_0[2];
            b2_0[0] = b0[0];
            b2_0[1] = b0[1];
            {
                uint32_t _addr_10 = static_cast<uint32_t>(x_addr + (unsigned int)((k_0 * 12 + v_0) * 4));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(b2_0[0]) : "memory");
            }
            {
                uint32_t _addr_11 = static_cast<uint32_t>(x_addr + (unsigned int)(((k_0 + 1) * 12 + v_0) * 4));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(b2_0[1]) : "memory");
            }
            unsigned int hi_pk[1];
            unsigned int lo_pk[1];
            float lo_v[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(b2_0[_lp*2 + 0], b2_0[_lp*2+1 + 0]));
                hi_pk[_lp] = *(uint32_t*)&_bf2;
            }
            lo_v[0] = b2_0[0] - __uint_as_float(hi_pk[0] << 16);
            lo_v[1] = b2_0[1] - __uint_as_float(hi_pk[0] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_v[_lp*2 + 0], lo_v[_lp*2+1 + 0]));
                lo_pk[_lp] = *(uint32_t*)&_bf2;
            }
            long long elem = (long long)((n * (p_cap - 1) * num_heads + h) * 2) * 16384ll + (long long)((v0 + v_0) * 128 + k_0);
            reinterpret_cast<unsigned*>(hpub + elem)[0] = reinterpret_cast<unsigned*>(hi_pk)[0];
            reinterpret_cast<unsigned*>(hpub + (elem + 16384ll))[0] = reinterpret_cast<unsigned*>(lo_pk)[0];
            int p_5 = tid + 256;
            int v_1 = p_5 / 64;
            int k_1 = p_5 % 64 * 2;
            float b2_1[2];
            b2_1[0] = b0[2];
            b2_1[1] = b0[3];
            {
                uint32_t _addr_12 = static_cast<uint32_t>(x_addr + (unsigned int)((k_1 * 12 + v_1) * 4));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(b2_1[0]) : "memory");
            }
            {
                uint32_t _addr_13 = static_cast<uint32_t>(x_addr + (unsigned int)(((k_1 + 1) * 12 + v_1) * 4));
                asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(b2_1[1]) : "memory");
            }
            unsigned int hi_pk_0[1];
            unsigned int lo_pk_0[1];
            float lo_v_0[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(b2_1[_lp*2 + 0], b2_1[_lp*2+1 + 0]));
                hi_pk_0[_lp] = *(uint32_t*)&_bf2;
            }
            lo_v_0[0] = b2_1[0] - __uint_as_float(hi_pk_0[0] << 16);
            lo_v_0[1] = b2_1[1] - __uint_as_float(hi_pk_0[0] & 4294901760u);
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_v_0[_lp*2 + 0], lo_v_0[_lp*2+1 + 0]));
                lo_pk_0[_lp] = *(uint32_t*)&_bf2;
            }
            long long elem_0 = (long long)((n * (p_cap - 1) * num_heads + h) * 2) * 16384ll + (long long)((v0 + v_1) * 128 + k_1);
            reinterpret_cast<unsigned*>(hpub + elem_0)[0] = reinterpret_cast<unsigned*>(hi_pk_0)[0];
            reinterpret_cast<unsigned*>(hpub + (elem_0 + 16384ll))[0] = reinterpret_cast<unsigned*>(lo_pk_0)[0];
            __syncthreads();
            if (tid == 0) {
                asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (((n * p_cap + 1) * num_heads + h) * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
            }
            float acc[32];
            float m4[4];
            float xx[8];
            #pragma unroll 1
            for (int s = 1; s < segs; s++) {
                unsigned int xc = x_addr + (unsigned int)((s - 1) % 2 * 6144);
                unsigned int xn = x_addr + (unsigned int)(s % 2 * 6144);
                if (s >= 2) {
                    int p_6 = tid;
                    {
                        float2 _v2_14 = *reinterpret_cast<const float2*>(bm + (((long long)(n * 2 * p_cap + s) * (long long)num_heads + (long long)h) * 16384ll + (long long)((v0 + p_6 / 64) * 128 + p_6 % 64 * 2)) + 0);
                        bpre[0] = _v2_14.x;
                        bpre[0 + 1] = _v2_14.y;
                    }
                    int p_7 = tid + 256;
                    {
                        float2 _v2_15 = *reinterpret_cast<const float2*>(bm + (((long long)(n * 2 * p_cap + s) * (long long)num_heads + (long long)h) * 16384ll + (long long)((v0 + p_7 / 64) * 128 + p_7 % 64 * 2)) + 0);
                        bpre[2] = _v2_15.x;
                        bpre[2 + 1] = _v2_15.y;
                    }
                }
                #pragma unroll
                for (int a = 0; a < 32; a++) {
                    acc[a] = 0.0f;
                }
                #pragma unroll 1
                for (int q = 0; q < 4; q++) {
                    int _stage_m_full = (int)((unsigned int)((s - 1) * 4 + q) % 4);
                    int _parity_m_full = (int)((unsigned int)((s - 1) * 4 + q) / 4 % 2);
                    mbarrier_wait(m_full_addr + (_stage_m_full) * 8, _parity_m_full);
                    unsigned int buf = ring_addr + (unsigned int)(((s - 1) * 4 + q) % 4 * 16384);
                    #pragma unroll
                    for (int rr = 0; rr < 4; rr++) {
                        int rl = rq * 4 + rr;
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&m4[0])), "=r"(*reinterpret_cast<uint32_t*>(&m4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&m4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&m4[(0) + 3]))
                            : "r"(buf + (unsigned int)((rl * 128 + 4 * lane) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&xx[0])), "=r"(*reinterpret_cast<uint32_t*>(&xx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xx[(0) + 3]))
                            : "r"(xc + (unsigned int)(((q * 32 + rl) * 12 + vg * 8) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&xx[4])), "=r"(*reinterpret_cast<uint32_t*>(&xx[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xx[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xx[(4) + 3]))
                            : "r"(xc + (unsigned int)(((q * 32 + rl) * 12 + vg * 8 + 4) * 4)));
                        acc[0] = __fmaf_rn(m4[0], xx[0], acc[0]);
                        acc[1] = __fmaf_rn(m4[0], xx[1], acc[1]);
                        acc[2] = __fmaf_rn(m4[0], xx[2], acc[2]);
                        acc[3] = __fmaf_rn(m4[0], xx[3], acc[3]);
                        acc[4] = __fmaf_rn(m4[0], xx[4], acc[4]);
                        acc[5] = __fmaf_rn(m4[0], xx[5], acc[5]);
                        acc[6] = __fmaf_rn(m4[0], xx[6], acc[6]);
                        acc[7] = __fmaf_rn(m4[0], xx[7], acc[7]);
                        acc[8] = __fmaf_rn(m4[1], xx[0], acc[8]);
                        acc[9] = __fmaf_rn(m4[1], xx[1], acc[9]);
                        acc[10] = __fmaf_rn(m4[1], xx[2], acc[10]);
                        acc[11] = __fmaf_rn(m4[1], xx[3], acc[11]);
                        acc[12] = __fmaf_rn(m4[1], xx[4], acc[12]);
                        acc[13] = __fmaf_rn(m4[1], xx[5], acc[13]);
                        acc[14] = __fmaf_rn(m4[1], xx[6], acc[14]);
                        acc[15] = __fmaf_rn(m4[1], xx[7], acc[15]);
                        acc[16] = __fmaf_rn(m4[2], xx[0], acc[16]);
                        acc[17] = __fmaf_rn(m4[2], xx[1], acc[17]);
                        acc[18] = __fmaf_rn(m4[2], xx[2], acc[18]);
                        acc[19] = __fmaf_rn(m4[2], xx[3], acc[19]);
                        acc[20] = __fmaf_rn(m4[2], xx[4], acc[20]);
                        acc[21] = __fmaf_rn(m4[2], xx[5], acc[21]);
                        acc[22] = __fmaf_rn(m4[2], xx[6], acc[22]);
                        acc[23] = __fmaf_rn(m4[2], xx[7], acc[23]);
                        acc[24] = __fmaf_rn(m4[3], xx[0], acc[24]);
                        acc[25] = __fmaf_rn(m4[3], xx[1], acc[25]);
                        acc[26] = __fmaf_rn(m4[3], xx[2], acc[26]);
                        acc[27] = __fmaf_rn(m4[3], xx[3], acc[27]);
                        acc[28] = __fmaf_rn(m4[3], xx[4], acc[28]);
                        acc[29] = __fmaf_rn(m4[3], xx[5], acc[29]);
                        acc[30] = __fmaf_rn(m4[3], xx[6], acc[30]);
                        acc[31] = __fmaf_rn(m4[3], xx[7], acc[31]);
                    }
                    __syncthreads();
                    if (q == 0 && s >= 2) {
                        if (tid == 0) {
                            asm volatile("red.async.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(flags) + (((n * p_cap + s) * num_heads + h) * 2))), "r"(static_cast<unsigned int>(1)) : "memory");
                        }
                    }
                    if (tid == 0) {
                        int cn = (s - 1) * 4 + q + 4;
                        if (cn < nchunks) {
                            long long src_n = ((long long)((n * 2 + 1) * p_cap + (1 + cn / 4)) * (long long)num_heads + (long long)h) * 16384ll + (long long)(cn % 4 * 32 * 128);
                            mbarrier_arrive_expect_tx(m_full_addr + ((int)((unsigned int)((s - 1) * 4 + q + 4) % 4)) * 8, 16384);
                            cp_async_bulk_gmem2smem(ring_addr + (unsigned int)(cn % 4 * 16384), bm + src_n, 16384, m_full_addr + ((int)((unsigned int)((s - 1) * 4 + q + 4) % 4)) * 8);
                        }
                    }
                }
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(part_addr + (unsigned int)(((rq * 8 + vg * 8) * 128 + 4 * lane) * 4)), "f"(acc[0]), "f"(acc[8]), "f"(acc[16]), "f"(acc[24]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(part_addr + (unsigned int)(((rq * 8 + vg * 8 + 1) * 128 + 4 * lane) * 4)), "f"(acc[1]), "f"(acc[9]), "f"(acc[17]), "f"(acc[25]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(part_addr + (unsigned int)(((rq * 8 + vg * 8 + 2) * 128 + 4 * lane) * 4)), "f"(acc[2]), "f"(acc[10]), "f"(acc[18]), "f"(acc[26]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(part_addr + (unsigned int)(((rq * 8 + vg * 8 + 3) * 128 + 4 * lane) * 4)), "f"(acc[3]), "f"(acc[11]), "f"(acc[19]), "f"(acc[27]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(part_addr + (unsigned int)(((rq * 8 + vg * 8 + 4) * 128 + 4 * lane) * 4)), "f"(acc[4]), "f"(acc[12]), "f"(acc[20]), "f"(acc[28]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(part_addr + (unsigned int)(((rq * 8 + vg * 8 + 5) * 128 + 4 * lane) * 4)), "f"(acc[5]), "f"(acc[13]), "f"(acc[21]), "f"(acc[29]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(part_addr + (unsigned int)(((rq * 8 + vg * 8 + 6) * 128 + 4 * lane) * 4)), "f"(acc[6]), "f"(acc[14]), "f"(acc[22]), "f"(acc[30]) : "memory");
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(part_addr + (unsigned int)(((rq * 8 + vg * 8 + 7) * 128 + 4 * lane) * 4)), "f"(acc[7]), "f"(acc[15]), "f"(acc[23]), "f"(acc[31]) : "memory");
                __syncthreads();
                int p_8 = tid;
                int v_2 = p_8 / 64;
                int k_2 = p_8 % 64 * 2;
                float sums_0[2];
                float pr_0[2];
                sums_0[0] = 0.0f;
                sums_0[1] = 0.0f;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_0[(0) + 1]))
                    : "r"(part_addr + (unsigned int)((v_2 * 128 + k_2) * 4)));
                sums_0[0] = sums_0[0] + pr_0[0];
                sums_0[1] = sums_0[1] + pr_0[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_0[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((8 + v_2) * 128 + k_2) * 4)));
                sums_0[0] = sums_0[0] + pr_0[0];
                sums_0[1] = sums_0[1] + pr_0[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_0[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((16 + v_2) * 128 + k_2) * 4)));
                sums_0[0] = sums_0[0] + pr_0[0];
                sums_0[1] = sums_0[1] + pr_0[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_0[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((24 + v_2) * 128 + k_2) * 4)));
                sums_0[0] = sums_0[0] + pr_0[0];
                sums_0[1] = sums_0[1] + pr_0[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_0[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((32 + v_2) * 128 + k_2) * 4)));
                sums_0[0] = sums_0[0] + pr_0[0];
                sums_0[1] = sums_0[1] + pr_0[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_0[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((40 + v_2) * 128 + k_2) * 4)));
                sums_0[0] = sums_0[0] + pr_0[0];
                sums_0[1] = sums_0[1] + pr_0[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_0[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((48 + v_2) * 128 + k_2) * 4)));
                sums_0[0] = sums_0[0] + pr_0[0];
                sums_0[1] = sums_0[1] + pr_0[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_0[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((56 + v_2) * 128 + k_2) * 4)));
                sums_0[0] = sums_0[0] + pr_0[0];
                sums_0[1] = sums_0[1] + pr_0[1];
                sums_0[0] = sums_0[0] + bpre[0];
                sums_0[1] = sums_0[1] + bpre[1];
                if (s < segs - 1) {
                    {
                        uint32_t _addr_16 = static_cast<uint32_t>(xn + (unsigned int)((k_2 * 12 + v_2) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(sums_0[0]) : "memory");
                    }
                    {
                        uint32_t _addr_17 = static_cast<uint32_t>(xn + (unsigned int)(((k_2 + 1) * 12 + v_2) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(sums_0[1]) : "memory");
                    }
                    unsigned int hi_pk_1[1];
                    unsigned int lo_pk_1[1];
                    float lo_v_1[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sums_0[_lp*2 + 0], sums_0[_lp*2+1 + 0]));
                        hi_pk_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    lo_v_1[0] = sums_0[0] - __uint_as_float(hi_pk_1[0] << 16);
                    lo_v_1[1] = sums_0[1] - __uint_as_float(hi_pk_1[0] & 4294901760u);
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_v_1[_lp*2 + 0], lo_v_1[_lp*2+1 + 0]));
                        lo_pk_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    long long elem_1 = (long long)(((n * (p_cap - 1) + s) * num_heads + h) * 2) * 16384ll + (long long)((v0 + v_2) * 128 + k_2);
                    reinterpret_cast<unsigned*>(hpub + elem_1)[0] = reinterpret_cast<unsigned*>(hi_pk_1)[0];
                    reinterpret_cast<unsigned*>(hpub + (elem_1 + 16384ll))[0] = reinterpret_cast<unsigned*>(lo_pk_1)[0];
                } else {
                    {
                        float2 _v2 = make_float2(sums_0[0 + 0], sums_0[0 + 1]);
                        *reinterpret_cast<float2*>((output_state + ((long long)n * output_state_stride_slot + (long long)((h * 128 + v0 + v_2) * 128 + k_2))) + 0) = _v2;
                    }
                }
                int p_9 = tid + 256;
                int v_3 = p_9 / 64;
                int k_3 = p_9 % 64 * 2;
                float sums_1[2];
                float pr_1[2];
                sums_1[0] = 0.0f;
                sums_1[1] = 0.0f;
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_1[(0) + 1]))
                    : "r"(part_addr + (unsigned int)((v_3 * 128 + k_3) * 4)));
                sums_1[0] = sums_1[0] + pr_1[0];
                sums_1[1] = sums_1[1] + pr_1[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_1[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((8 + v_3) * 128 + k_3) * 4)));
                sums_1[0] = sums_1[0] + pr_1[0];
                sums_1[1] = sums_1[1] + pr_1[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_1[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((16 + v_3) * 128 + k_3) * 4)));
                sums_1[0] = sums_1[0] + pr_1[0];
                sums_1[1] = sums_1[1] + pr_1[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_1[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((24 + v_3) * 128 + k_3) * 4)));
                sums_1[0] = sums_1[0] + pr_1[0];
                sums_1[1] = sums_1[1] + pr_1[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_1[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((32 + v_3) * 128 + k_3) * 4)));
                sums_1[0] = sums_1[0] + pr_1[0];
                sums_1[1] = sums_1[1] + pr_1[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_1[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((40 + v_3) * 128 + k_3) * 4)));
                sums_1[0] = sums_1[0] + pr_1[0];
                sums_1[1] = sums_1[1] + pr_1[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_1[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((48 + v_3) * 128 + k_3) * 4)));
                sums_1[0] = sums_1[0] + pr_1[0];
                sums_1[1] = sums_1[1] + pr_1[1];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pr_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pr_1[(0) + 1]))
                    : "r"(part_addr + (unsigned int)(((56 + v_3) * 128 + k_3) * 4)));
                sums_1[0] = sums_1[0] + pr_1[0];
                sums_1[1] = sums_1[1] + pr_1[1];
                sums_1[0] = sums_1[0] + bpre[2];
                sums_1[1] = sums_1[1] + bpre[3];
                if (s < segs - 1) {
                    {
                        uint32_t _addr_18 = static_cast<uint32_t>(xn + (unsigned int)((k_3 * 12 + v_3) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_18), "f"(sums_1[0]) : "memory");
                    }
                    {
                        uint32_t _addr_19 = static_cast<uint32_t>(xn + (unsigned int)(((k_3 + 1) * 12 + v_3) * 4));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(sums_1[1]) : "memory");
                    }
                    unsigned int hi_pk_2[1];
                    unsigned int lo_pk_2[1];
                    float lo_v_2[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sums_1[_lp*2 + 0], sums_1[_lp*2+1 + 0]));
                        hi_pk_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    lo_v_2[0] = sums_1[0] - __uint_as_float(hi_pk_2[0] << 16);
                    lo_v_2[1] = sums_1[1] - __uint_as_float(hi_pk_2[0] & 4294901760u);
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo_v_2[_lp*2 + 0], lo_v_2[_lp*2+1 + 0]));
                        lo_pk_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    long long elem_2 = (long long)(((n * (p_cap - 1) + s) * num_heads + h) * 2) * 16384ll + (long long)((v0 + v_3) * 128 + k_3);
                    reinterpret_cast<unsigned*>(hpub + elem_2)[0] = reinterpret_cast<unsigned*>(hi_pk_2)[0];
                    reinterpret_cast<unsigned*>(hpub + (elem_2 + 16384ll))[0] = reinterpret_cast<unsigned*>(lo_pk_2)[0];
                } else {
                    {
                        float2 _v2 = make_float2(sums_1[0 + 0], sums_1[0 + 1]);
                        *reinterpret_cast<float2*>((output_state + ((long long)n * output_state_stride_slot + (long long)((h * 128 + v0 + v_3) * 128 + k_3))) + 0) = _v2;
                    }
                }
                __syncthreads();
            }
        }
        if (tid == 0) {
            uint32_t seen_ld;
            asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(seen_ld) : "l"(ticket) : "memory");
            unsigned int seen = seen_ld;
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(role_cell_addr + 12), "r"(seen) : "memory");
        }
        __syncthreads();
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&cell[0])) : "r"(role_cell_addr + 12));
        if (cell[0] != 0) {
            if ((int)cell[0] < n_compose) {
                join = 0;
            }
        }
    }
    if (join == 1) {
        int _min_20 = (((total_tokens + 127) / 128 * (p_cap - 1) * 2 * num_heads / (p_cap * 2 * gridDim.x)) < (8) ? ((total_tokens + 127) / 128 * (p_cap - 1) * 2 * num_heads / (p_cap * 2 * gridDim.x)) : (8));
        int run = ((_min_20) > (1) ? (_min_20) : (1));
        if (tid == 0) {
            unsigned int claim0 = atomicAdd(&ticket[1], 1);
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(role_cell_addr + 4), "r"(claim0) : "memory");
        }
        __syncthreads();
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&cell[0])) : "r"(role_cell_addr + 4));
        int unit = (int)cell[0];
        int ws = 1;
        int wn = 0;
        int wbase = 0;
        int wv = 0;
        int wl[5];
        int kt = 0;
        int ku = 0;
        float accv[32];
        float ov[32];
        unsigned int pk[16];
        int trow = warp % 4 * 32 + lane;
        int vq = warp / 4 * 32;
        int active = 1;
        while (active == 1) {
            int seg_a = 0;
            int seg_e = 0;
            int nblk = 0;
            int found = 0;
            int gs = 0;
            int gn = 0;
            int gbase = 0;
            found = 0;
            while (found == 0) {
                if (wv == 0) {
                    if (ws >= p_cap) {
                        found = 2;
                    } else {
                        int q_1 = wn + lane;
                        int l_s = ws + q_1 / num_seqs;
                        int l_n = q_1 % num_seqs;
                        wl[0] = 0;
                        wl[1] = 0;
                        wl[2] = 0;
                        wl[3] = 0;
                        if (l_s < p_cap) {
                            {
                                uint32_t _scalar_bits_21;
                                asm volatile("ld.global.nc.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_21) : "l"((const void*)(cu_seqlens + (l_n))) : "memory");
                                cuv[0] = (int32_t)_scalar_bits_21;
                            }
                            {
                                uint32_t _scalar_bits_22;
                                asm volatile("ld.global.nc.b32 %0, [%1];"
                                    : "=r"(_scalar_bits_22) : "l"((const void*)(cu_seqlens + (l_n + 1))) : "memory");
                                cuv[1] = (int32_t)_scalar_bits_22;
                            }
                            int _max_23 = ((cuv[0]) > (0) ? (cuv[0]) : (0));
                            int g_start = ((_max_23) < (total_tokens) ? (_max_23) : (total_tokens));
                            int _max_25 = ((cuv[1]) > (g_start) ? (cuv[1]) : (g_start));
                            int _min_24 = ((_max_25) < (total_tokens) ? (_max_25) : (total_tokens));
                            int g_len = _min_24 - g_start;
                            int g_chunks = (g_len + 63) / 64;
                            int _min_26 = ((g_chunks / c_min) < (p_cap) ? (g_chunks / c_min) : (p_cap));
                            int g_segs = ((_min_26) > (1) ? (_min_26) : (1));
                            if (l_s < g_segs) {
                                wl[0] = g_start + 64 * (l_s * g_chunks / g_segs);
                                int _min_27 = ((64 * ((l_s + 1) * g_chunks / g_segs)) < (g_len) ? (64 * ((l_s + 1) * g_chunks / g_segs)) : (g_len));
                                wl[1] = g_start + _min_27;
                                wl[2] = (wl[1] - wl[0] + 127) / 128;
                                wl[3] = 2 * num_heads * ((wl[2] + run - 1) / run);
                            }
                        }
                        int _warp_scan_sum_28 = wl[3];
                        {
                            uint32_t _warp_scan_sum_28_scan = (uint32_t)_warp_scan_sum_28;
                            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_28_scan, 1, 32); if (lane >= 1) { _warp_scan_sum_28_scan += peer; } }
                            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_28_scan, 2, 32); if (lane >= 2) { _warp_scan_sum_28_scan += peer; } }
                            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_28_scan, 4, 32); if (lane >= 4) { _warp_scan_sum_28_scan += peer; } }
                            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_28_scan, 8, 32); if (lane >= 8) { _warp_scan_sum_28_scan += peer; } }
                            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_28_scan, 16, 32); if (lane >= 16) { _warp_scan_sum_28_scan += peer; } }
                            _warp_scan_sum_28 = (int)_warp_scan_sum_28_scan;
                        }
                        wl[4] = _warp_scan_sum_28;
                        wv = 1;
                    }
                }
                if (found == 0) {
                    int total = __shfl_sync(0xFFFFFFFF, wl[4], 31);
                    if (unit < wbase + total) {
                        unsigned int _warp_vote_29 = __ballot_sync(0xFFFFFFFF, unit >= wbase + wl[4]);
                        int passed = (int)(unsigned int)__popc(_warp_vote_29);
                        found = 1;
                        int _shfl_sync_30 = __shfl_sync(0xFFFFFFFF, wl[0], passed);
                        seg_a = _shfl_sync_30;
                        int _shfl_sync_31 = __shfl_sync(0xFFFFFFFF, wl[1], passed);
                        seg_e = _shfl_sync_31;
                        int _shfl_sync_32 = __shfl_sync(0xFFFFFFFF, wl[2], passed);
                        nblk = _shfl_sync_32;
                        int _shfl_sync_33 = __shfl_sync(0xFFFFFFFF, wl[4] - wl[3], passed);
                        gbase = wbase + _shfl_sync_33;
                        gs = ws + (wn + passed) / num_seqs;
                        gn = (wn + passed) % num_seqs;
                    } else {
                        wbase += total;
                        ws += (wn + 32) / num_seqs;
                        wn = (wn + 32) % num_seqs;
                        wv = 0;
                    }
                }
            }
            if (found == 2) {
                active = 0;
            } else {
                int s_1 = gs;
                int n_1 = gn;
                int runs = (nblk + run - 1) / run;
                int hj = (unit - gbase) / runs;
                int h_1 = hj / 2;
                int j = hj % 2;
                int b0_1 = (unit - gbase) % runs * run;
                int n_tiles = ((run) < (nblk - b0_1) ? (run) : (nblk - b0_1));
                int flag = ((n_1 * p_cap + s_1) * num_heads + h_1) * 2;
                unsigned int claim1 = 0;
                unsigned int consumed = 0;
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(in_full_addr + ((int)((unsigned int)kt % 2)) * 8, 32768);
                        tma_3d_gmem2smem(c_addr + (unsigned int)(kt % 2 * 32768), (&c_tma), 0, h_1, seg_a + 128 * b0_1, in_full_addr + ((int)((unsigned int)kt % 2)) * 8);
                        tma_3d_gmem2smem(c_addr + (unsigned int)(kt % 2 * 32768 + 16384), (&c_tma), 64, h_1, seg_a + 128 * b0_1, in_full_addr + ((int)((unsigned int)kt % 2)) * 8);
                    }
                }
                if (n_tiles > 1) {
                    if (warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(in_full_addr + ((int)((unsigned int)(kt + 1) % 2)) * 8, 32768);
                            tma_3d_gmem2smem(c_addr + (unsigned int)((kt + 1) % 2 * 32768), (&c_tma), 0, h_1, seg_a + 128 * (b0_1 + 1), in_full_addr + ((int)((unsigned int)(kt + 1) % 2)) * 8);
                            tma_3d_gmem2smem(c_addr + (unsigned int)((kt + 1) % 2 * 32768 + 16384), (&c_tma), 64, h_1, seg_a + 128 * (b0_1 + 1), in_full_addr + ((int)((unsigned int)(kt + 1) % 2)) * 8);
                        }
                    }
                }
                if (warp == 0) {
                    if (elect_sync()) {
                        {
                        unsigned int _acquire_observed;
                        do {
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(flags) + (flag))) : "memory");
                        } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(16)) >= static_cast<unsigned int>(2147483648u));
                        }
                        asm volatile("fence.proxy.async.global;" ::: "memory");
                        int hrow = ((n_1 * (p_cap - 1) + s_1 - 1) * num_heads + h_1) * 2;
                        mbarrier_arrive_expect_tx(h_full_addr, 32768);
                        tma_3d_gmem2smem(h_hi_addr + 0, (&hpub_tma), 0, 64 * j, hrow, h_full_addr);
                        tma_3d_gmem2smem(h_hi_addr + 8192, (&hpub_tma), 64, 64 * j, hrow, h_full_addr);
                        tma_3d_gmem2smem(h_hi_addr + 16384, (&hpub_tma), 0, 64 * j, hrow + 1, h_full_addr);
                        tma_3d_gmem2smem(h_hi_addr + 24576, (&hpub_tma), 64, 64 * j, hrow + 1, h_full_addr);
                    }
                }
                int v0_1 = 64 * j + vq;
                #pragma unroll 1
                for (int i = 0; i < n_tiles; i++) {
                    int t0 = seg_a + 128 * (b0_1 + i);
                    int tk = t0 + trow;
                    if (i + 2 < n_tiles) {
                        int row = t0 + 256 + tid / 2;
                        long long head_row = ((long long)row * (long long)num_heads + (long long)h_1) * 128ll;
                        if (row < seg_e) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(c_rows + (head_row + (long long)(tid % 2 * 64))))); }
                        if (row < seg_e) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(o_loc + (head_row + (long long)(64 * j + tid % 2 * 32))))); }
                    }
                    int _min_34 = ((tk) < (seg_e - 1) ? (tk) : (seg_e - 1));
                    long long o_at = ((long long)_min_34 * (long long)num_heads + (long long)h_1) * 128ll + (long long)v0_1;
                    {
                        unsigned _ldv8_35_0;
                        unsigned _ldv8_35_1;
                        unsigned _ldv8_35_2;
                        unsigned _ldv8_35_3;
                        unsigned _ldv8_35_4;
                        unsigned _ldv8_35_5;
                        unsigned _ldv8_35_6;
                        unsigned _ldv8_35_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_35_0), "=r"(_ldv8_35_1), "=r"(_ldv8_35_2), "=r"(_ldv8_35_3), "=r"(_ldv8_35_4), "=r"(_ldv8_35_5), "=r"(_ldv8_35_6), "=r"(_ldv8_35_7) : "l"((const void*)(o_loc + (o_at + 0ll) + (0))) : "memory");
                        ov[0 + 0] = __uint_as_float(_ldv8_35_0);
                        ov[0 + 1] = __uint_as_float(_ldv8_35_1);
                        ov[0 + 2] = __uint_as_float(_ldv8_35_2);
                        ov[0 + 3] = __uint_as_float(_ldv8_35_3);
                        ov[0 + 4] = __uint_as_float(_ldv8_35_4);
                        ov[0 + 5] = __uint_as_float(_ldv8_35_5);
                        ov[0 + 6] = __uint_as_float(_ldv8_35_6);
                        ov[0 + 7] = __uint_as_float(_ldv8_35_7);
                    }
                    {
                        unsigned _ldv8_36_0;
                        unsigned _ldv8_36_1;
                        unsigned _ldv8_36_2;
                        unsigned _ldv8_36_3;
                        unsigned _ldv8_36_4;
                        unsigned _ldv8_36_5;
                        unsigned _ldv8_36_6;
                        unsigned _ldv8_36_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_36_0), "=r"(_ldv8_36_1), "=r"(_ldv8_36_2), "=r"(_ldv8_36_3), "=r"(_ldv8_36_4), "=r"(_ldv8_36_5), "=r"(_ldv8_36_6), "=r"(_ldv8_36_7) : "l"((const void*)(o_loc + (o_at + 8ll) + (0))) : "memory");
                        ov[8 + 0] = __uint_as_float(_ldv8_36_0);
                        ov[8 + 1] = __uint_as_float(_ldv8_36_1);
                        ov[8 + 2] = __uint_as_float(_ldv8_36_2);
                        ov[8 + 3] = __uint_as_float(_ldv8_36_3);
                        ov[8 + 4] = __uint_as_float(_ldv8_36_4);
                        ov[8 + 5] = __uint_as_float(_ldv8_36_5);
                        ov[8 + 6] = __uint_as_float(_ldv8_36_6);
                        ov[8 + 7] = __uint_as_float(_ldv8_36_7);
                    }
                    {
                        unsigned _ldv8_37_0;
                        unsigned _ldv8_37_1;
                        unsigned _ldv8_37_2;
                        unsigned _ldv8_37_3;
                        unsigned _ldv8_37_4;
                        unsigned _ldv8_37_5;
                        unsigned _ldv8_37_6;
                        unsigned _ldv8_37_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_37_0), "=r"(_ldv8_37_1), "=r"(_ldv8_37_2), "=r"(_ldv8_37_3), "=r"(_ldv8_37_4), "=r"(_ldv8_37_5), "=r"(_ldv8_37_6), "=r"(_ldv8_37_7) : "l"((const void*)(o_loc + (o_at + 16ll) + (0))) : "memory");
                        ov[16 + 0] = __uint_as_float(_ldv8_37_0);
                        ov[16 + 1] = __uint_as_float(_ldv8_37_1);
                        ov[16 + 2] = __uint_as_float(_ldv8_37_2);
                        ov[16 + 3] = __uint_as_float(_ldv8_37_3);
                        ov[16 + 4] = __uint_as_float(_ldv8_37_4);
                        ov[16 + 5] = __uint_as_float(_ldv8_37_5);
                        ov[16 + 6] = __uint_as_float(_ldv8_37_6);
                        ov[16 + 7] = __uint_as_float(_ldv8_37_7);
                    }
                    {
                        unsigned _ldv8_38_0;
                        unsigned _ldv8_38_1;
                        unsigned _ldv8_38_2;
                        unsigned _ldv8_38_3;
                        unsigned _ldv8_38_4;
                        unsigned _ldv8_38_5;
                        unsigned _ldv8_38_6;
                        unsigned _ldv8_38_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_38_0), "=r"(_ldv8_38_1), "=r"(_ldv8_38_2), "=r"(_ldv8_38_3), "=r"(_ldv8_38_4), "=r"(_ldv8_38_5), "=r"(_ldv8_38_6), "=r"(_ldv8_38_7) : "l"((const void*)(o_loc + (o_at + 24ll) + (0))) : "memory");
                        ov[24 + 0] = __uint_as_float(_ldv8_38_0);
                        ov[24 + 1] = __uint_as_float(_ldv8_38_1);
                        ov[24 + 2] = __uint_as_float(_ldv8_38_2);
                        ov[24 + 3] = __uint_as_float(_ldv8_38_3);
                        ov[24 + 4] = __uint_as_float(_ldv8_38_4);
                        ov[24 + 5] = __uint_as_float(_ldv8_38_5);
                        ov[24 + 6] = __uint_as_float(_ldv8_38_6);
                        ov[24 + 7] = __uint_as_float(_ldv8_38_7);
                    }
                    if (warp == 0) {
                        int _stage_in_full = (int)((unsigned int)kt % 2);
                        int _parity_in_full = (int)((unsigned int)kt / 2 % 2);
                        mbarrier_wait(in_full_addr + (_stage_in_full) * 8, _parity_in_full);
                        if (i == 0) {
                            int _parity_h_full = (int)((unsigned int)ku % 2);
                            mbarrier_wait(h_full_addr, _parity_h_full);
                            if (tid == 0) {
                                uint32_t _atomic_fetch_inc_39;
                                asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"
                                    : "=r"(_atomic_fetch_inc_39) : "l"(&flags[flag + 1]), "r"(static_cast<uint32_t>((unsigned int)(2 * runs - 1))) : "memory");
                                consumed = _atomic_fetch_inc_39;
                            }
                        }
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int a_0 = make_warp_uniform(((c_addr) >> 4) & 0x7FFF);
                        int a_1 = make_warp_uniform(((_c_1_addr) >> 4) & 0x7FFF);
                        int b_hi = make_warp_uniform(((h_hi_addr) >> 4) & 0x7FFF);
                        int b_lo = make_warp_uniform(((h_lo_addr) >> 4) & 0x7FFF);
#else
                        int a_0 = make_warp_uniform(((c_addr) >> 4) & 0x3FFF);
                        int a_1 = make_warp_uniform(((_c_1_addr) >> 4) & 0x3FFF);
                        int b_hi = make_warp_uniform(((h_hi_addr) >> 4) & 0x3FFF);
                        int b_lo = make_warp_uniform(((h_lo_addr) >> 4) & 0x3FFF);
#endif
                        if (elect_sync()) {
                            if (kt % 2 == 0) {
                                asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 135267472;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 1018;\n\t"
            "add.u32 blo, blo, 506;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(a_0), "r"(b_hi), "r"(tmem_acc), "r"(0) : "memory");
                                asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 135267472;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 1018;\n\t"
            "add.u32 blo, blo, 506;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(a_0), "r"(b_lo), "r"(tmem_acc), "r"(1) : "memory");
                            } else {
                                asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 135267472;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 1018;\n\t"
            "add.u32 blo, blo, 506;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(a_1), "r"(b_hi), "r"(tmem_acc), "r"(0) : "memory");
                                asm volatile(
            "{\n\t"
            ".reg .pred p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            ""
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 135267472;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 1018;\n\t"
            "add.u32 blo, blo, 506;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(a_1), "r"(b_lo), "r"(tmem_acc), "r"(1) : "memory");
                            }
                            tcgen05_commit(mma_done_addr);
                        }
                        int _parity_mma_done = (int)((unsigned int)kt % 2);
                        mbarrier_wait(mma_done_addr, _parity_mma_done);
                        if (i + 2 < n_tiles) {
                            if (warp == 0) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(in_full_addr + ((int)((unsigned int)(kt + 2) % 2)) * 8, 32768);
                                    tma_3d_gmem2smem(c_addr + (unsigned int)((kt + 2) % 2 * 32768), (&c_tma), 0, h_1, t0 + 256, in_full_addr + ((int)((unsigned int)(kt + 2) % 2)) * 8);
                                    tma_3d_gmem2smem(c_addr + (unsigned int)((kt + 2) % 2 * 32768 + 16384), (&c_tma), 64, h_1, t0 + 256, in_full_addr + ((int)((unsigned int)(kt + 2) % 2)) * 8);
                                }
                            }
                        }
                        if (i == n_tiles - 1) {
                            if (tid == 0) {
                                unsigned int _atomic_fetch_add_40 = atomicAdd(&ticket[1], 1);
                                claim1 = _atomic_fetch_add_40;
                            }
                        }
                    }
                    __syncthreads();
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(accv[0]), "=f"(accv[1]), "=f"(accv[2]), "=f"(accv[3]), "=f"(accv[4]), "=f"(accv[5]), "=f"(accv[6]), "=f"(accv[7]), "=f"(accv[8]), "=f"(accv[9]), "=f"(accv[10]), "=f"(accv[11]), "=f"(accv[12]), "=f"(accv[13]), "=f"(accv[14]), "=f"(accv[15]), "=f"(accv[16]), "=f"(accv[17]), "=f"(accv[18]), "=f"(accv[19]), "=f"(accv[20]), "=f"(accv[21]), "=f"(accv[22]), "=f"(accv[23]), "=f"(accv[24]), "=f"(accv[25]), "=f"(accv[26]), "=f"(accv[27]), "=f"(accv[28]), "=f"(accv[29]), "=f"(accv[30]), "=f"(accv[31])
                        : "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)vq));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    accv[0] = accv[0] + ov[0];
                    accv[1] = accv[1] + ov[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 0], accv[_lp*2+1 + 0]));
                        pk[_lp] = *(uint32_t*)&_bf2;
                    }
                    accv[2] = accv[2] + ov[2];
                    accv[3] = accv[3] + ov[3];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 2], accv[_lp*2+1 + 2]));
                        pk[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    accv[4] = accv[4] + ov[4];
                    accv[5] = accv[5] + ov[5];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 4], accv[_lp*2+1 + 4]));
                        pk[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    accv[6] = accv[6] + ov[6];
                    accv[7] = accv[7] + ov[7];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 6], accv[_lp*2+1 + 6]));
                        pk[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    accv[8] = accv[8] + ov[8];
                    accv[9] = accv[9] + ov[9];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 8], accv[_lp*2+1 + 8]));
                        pk[_lp + 4] = *(uint32_t*)&_bf2;
                    }
                    accv[10] = accv[10] + ov[10];
                    accv[11] = accv[11] + ov[11];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 10], accv[_lp*2+1 + 10]));
                        pk[_lp + 5] = *(uint32_t*)&_bf2;
                    }
                    accv[12] = accv[12] + ov[12];
                    accv[13] = accv[13] + ov[13];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 12], accv[_lp*2+1 + 12]));
                        pk[_lp + 6] = *(uint32_t*)&_bf2;
                    }
                    accv[14] = accv[14] + ov[14];
                    accv[15] = accv[15] + ov[15];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 14], accv[_lp*2+1 + 14]));
                        pk[_lp + 7] = *(uint32_t*)&_bf2;
                    }
                    accv[16] = accv[16] + ov[16];
                    accv[17] = accv[17] + ov[17];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 16], accv[_lp*2+1 + 16]));
                        pk[_lp + 8] = *(uint32_t*)&_bf2;
                    }
                    accv[18] = accv[18] + ov[18];
                    accv[19] = accv[19] + ov[19];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 18], accv[_lp*2+1 + 18]));
                        pk[_lp + 9] = *(uint32_t*)&_bf2;
                    }
                    accv[20] = accv[20] + ov[20];
                    accv[21] = accv[21] + ov[21];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 20], accv[_lp*2+1 + 20]));
                        pk[_lp + 10] = *(uint32_t*)&_bf2;
                    }
                    accv[22] = accv[22] + ov[22];
                    accv[23] = accv[23] + ov[23];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 22], accv[_lp*2+1 + 22]));
                        pk[_lp + 11] = *(uint32_t*)&_bf2;
                    }
                    accv[24] = accv[24] + ov[24];
                    accv[25] = accv[25] + ov[25];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 24], accv[_lp*2+1 + 24]));
                        pk[_lp + 12] = *(uint32_t*)&_bf2;
                    }
                    accv[26] = accv[26] + ov[26];
                    accv[27] = accv[27] + ov[27];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 26], accv[_lp*2+1 + 26]));
                        pk[_lp + 13] = *(uint32_t*)&_bf2;
                    }
                    accv[28] = accv[28] + ov[28];
                    accv[29] = accv[29] + ov[29];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 28], accv[_lp*2+1 + 28]));
                        pk[_lp + 14] = *(uint32_t*)&_bf2;
                    }
                    accv[30] = accv[30] + ov[30];
                    accv[31] = accv[31] + ov[31];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accv[_lp*2 + 30], accv[_lp*2+1 + 30]));
                        pk[_lp + 15] = *(uint32_t*)&_bf2;
                    }
                    long long out_at = ((long long)tk * (long long)num_heads + (long long)h_1) * 128ll + (long long)v0_1;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(output + (out_at + 0ll)), "r"(pk[(0) + 0]), "r"(pk[(0) + 1]), "r"(pk[(0) + 2]), "r"(pk[(0) + 3]), "r"((unsigned int)(tk < seg_e)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(output + (out_at + 8ll)), "r"(pk[(4) + 0]), "r"(pk[(4) + 1]), "r"(pk[(4) + 2]), "r"(pk[(4) + 3]), "r"((unsigned int)(tk < seg_e)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(output + (out_at + 16ll)), "r"(pk[(8) + 0]), "r"(pk[(8) + 1]), "r"(pk[(8) + 2]), "r"(pk[(8) + 3]), "r"((unsigned int)(tk < seg_e)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.b32 [%0], {%1, %2, %3, %4}; }" :: "l"(output + (out_at + 24ll)), "r"(pk[(12) + 0]), "r"(pk[(12) + 1]), "r"(pk[(12) + 2]), "r"(pk[(12) + 3]), "r"((unsigned int)(tk < seg_e)) : "memory");
                    if (i == n_tiles - 1) {
                        if (tid == 0) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(role_cell_addr + 4), "r"(claim1) : "memory");
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncthreads();
                    kt += 1;
                }
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&cell[0])) : "r"(role_cell_addr + 4));
                if (tid == 0) {
                    if (consumed == (unsigned int)(2 * runs - 1)) {
                        asm volatile("st.volatile.global.u32 [%0], %1;" :: "l"(&flags[flag]), "r"(0) : "memory");
                    }
                }
                unit = (int)cell[0];
                ku += 1;
            }
        }
    }
    if (tid == 0) {
        uint32_t done;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(done) : "l"(&ticket[2]), "r"(static_cast<uint32_t>((unsigned int)(gridDim.x - 1))) : "memory");
        if (done == (unsigned int)(gridDim.x - 1)) {
            *(reinterpret_cast<unsigned int*>(ticket + 1) + (0)) = 0;
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
    }
}

} // extern "C"
