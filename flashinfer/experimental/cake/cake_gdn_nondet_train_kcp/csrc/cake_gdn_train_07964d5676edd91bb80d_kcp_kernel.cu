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

#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_07964d5676edd91bb80d_kcp(float* __restrict__ gamma, float* __restrict__ rowscale, __nv_bfloat16* __restrict__ w, int* __restrict__ chunk_len, int num_heads, int* __restrict__ seq_chunk_start, int num_seqs, int* __restrict__ walk_seg, int* __restrict__ c_start)
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
    int* found = reinterpret_cast<int*>(smem_raw + 0);
    const int found_addr = smem + 0;

    // === Task calls (declaration order) ===
    int head = blockIdx.x % num_heads;
    int seg = walk_seg[blockIdx.x / num_heads];
    int c0 = seq_chunk_start[seg];
    int nc = seq_chunk_start[seg + 1] - c0;
    int z = -1;
    int seen = 0;
    #pragma unroll 1
    for (int base = 0; base < nc; base += 32) {
        int c = base + lane;
        float gm = 1.0f;
        if (c < nc) {
            gm = gamma[(long long)(c0 + c) * (long long)num_heads + (long long)head];
        }
        unsigned int cand = __ballot_sync(0xFFFFFFFF, c < nc && gm == 0.0f);
        int k = (int)(unsigned int)__popc(cand);
        if (seen + k > warp) {
            #pragma unroll 1
            for (int _drop = seen; _drop < warp; _drop++) {
                cand = cand & cand - 1;
            }
            int _find_first_set_0 = __ffs(cand);
            z = base + _find_first_set_0 - 1;
            break;
        }
        seen = seen + k;
    }
    int zr = -1;
    if (z >= 0) {
        int ln = chunk_len[c0 + z];
        long long r0 = (long long)(c0 + z) * 64ll;
        int dep = 0;
        unsigned int wv[4];
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            int t = lane + hh * 32;
            if (t < ln) {
                long long rt = (r0 + (long long)t) * (long long)num_heads + (long long)head;
                float rs = rowscale[rt];
                unsigned int bits = 0;
                #pragma unroll
                for (int q = 0; q < 16; q++) {
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + rt * 128ll + (long long)(q * 8));
                        uint4* _vdst_1 = reinterpret_cast<uint4*>(&wv[0]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_vdst_1[_blk].x), "=r"(_vdst_1[_blk].y), "=r"(_vdst_1[_blk].z), "=r"(_vdst_1[_blk].w) : "l"((const void*)(_vptr_1 + _blk)) : "memory");
                        }
                    }
                    bits = bits | (wv[0] | wv[1] | (wv[2] | wv[3]));
                }
                if (!(rs == 0.0f) && (bits & 2147450879) != 0) {
                    dep = 1;
                }
            }
        }
        unsigned int _warp_vote_2 = __ballot_sync(0xFFFFFFFF, dep != 0);
        if (_warp_vote_2 == 0) {
            zr = z;
        }
    }
    if (lane == 0) {
        found[warp] = zr;
    }
    __syncthreads();
    if (tid == 0) {
        int cs = nc - 1;
        int zj_0 = found[3];
        if (zj_0 >= 0) {
            cs = zj_0;
        }
        int zj_1 = found[2];
        if (zj_1 >= 0) {
            cs = zj_1;
        }
        int zj_2 = found[1];
        if (zj_2 >= 0) {
            cs = zj_2;
        }
        int zj_3 = found[0];
        if (zj_3 >= 0) {
            cs = zj_3;
        }
        c_start[(long long)seg * (long long)num_heads + (long long)head] = cs;
    }
}

} // extern "C"
