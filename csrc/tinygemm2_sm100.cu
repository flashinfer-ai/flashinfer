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

// Generated SM100/SM103 port of the TensorRT-LLM tinygemm2 kernel
// (csrc/tinygemm2.cu) with bit-identical outputs, in one translation unit
// (kernel family + TVM-FFI binding) following the csrc/tinygemm2.cu
// convention.
//
// One kernel template, tinygemm2_kernel<STAGES, USE_PDL>, is instantiated for
// the three pipeline ring depths STAGES in {4, 8, 16} and for PDL off/on (all
// six instantiations are reached from the binding's dispatcher). Each compiles
// to the same code as the formerly separate per-variant kernel: every
// STAGES-dependent quantity (shared-memory layout, mbarrier count, ring slot
// and phase arithmetic, drain-loop trip count) is a constant expression of
// STAGES, and the PDL griddepcontrol pair sits under `if constexpr (USE_PDL)`. The ring depth is selected in the binding from the
// problem shape and the device SM count (SelectStages).
//
// Both 128-byte TMA descriptors ride in one trailing by-value __grid_constant__
// pack (TensorMapPack<2>: maps[0] = weight, maps[1] = activation).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace tinygemm2_sm100 {

// clang-format off

struct __align__(128) TensorMapBytes { uint64_t opaque[16]; };
template <int N>
struct __align__(128) TensorMapPack { TensorMapBytes maps[N]; };

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

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

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
}

__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ unsigned int float_bits(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}

// Fixed tile geometry (TensorRT-LLM tinygemm2 template constants
// WARP_TILE_M=16, TILE_N=8, TILE_K=64, STAGE_UNROLL=4).
constexpr int kTileM = 16;  // output-features tile
constexpr int kTileN = 8;   // batch tile
constexpr int kTileK = 64;  // reduction tile (one TMA box)
constexpr int kStageUnroll = 4;
constexpr int kThreads = 384;
// One loader K loop covers four loader warps x kStageUnroll boxes = 1024
// reduction elements.
constexpr int kKPerLoop = 4 * kTileK * kStageUnroll;

// Shared-memory layout of one STAGES instance: 3 x STAGES mbarriers at offset
// 0 (padded to 1 KiB), the weight ring, the activation ring, the 2 KiB split-K
// reduction scratch (float4[128]) and the 128-byte bias tile.
constexpr int kWtSubBytes = kTileM * kTileK * 2;   // 2048
constexpr int kActSubBytes = kTileN * kTileK * 2;  // 1024
constexpr int kRedBytes = 128 * 16;
constexpr int kBiasBytes = 128;
constexpr int kSmemWtOff = 1024;
constexpr int SmemActOff(int stages) { return kSmemWtOff + stages * kStageUnroll * kWtSubBytes; }
constexpr int SmemRedOff(int stages) { return SmemActOff(stages) + stages * kStageUnroll * kActSubBytes; }
constexpr int SmemBiasOff(int stages) { return SmemRedOff(stages) + kRedBytes; }
constexpr int SmemBytes(int stages) { return SmemBiasOff(stages) + kBiasBytes; }
static_assert(SmemBytes(4) == 52352 && SmemBytes(8) == 101504 && SmemBytes(16) == 199808,
              "shared-memory footprint of a ring depth changed");
static_assert(3 * 16 * 8 <= kSmemWtOff, "mbarriers must fit below the weight ring");

// out[n, m] = sum_k act[n, k] * wt[m, k] + bias[m] for one 16 x 8 output tile
// per CTA. Warps 0-3 compute (warp w consumes ring slots congruent to w mod 4
// and the four partial sums are reduced through smem_red), warps 4-7 load
// weights, warps 8-11 load activations; one elected lane per loader warp
// issues the TMA for its slots. Ring slot s holds kStageUnroll boxes; the
// STAGES/4 ring passes per loader warp use the mbarrier phase bit.
template <int STAGES, bool USE_PDL>
__global__ __launch_bounds__(384, 1) void
tinygemm2_kernel(__nv_bfloat16* __restrict__ output, __nv_bfloat16* __restrict__ bias, int M, int N, int K, __grid_constant__ TensorMapPack<2> const tma_params)
{
    static_assert(STAGES == 4 || STAGES == 8 || STAGES == 16, "ring depth must be 4, 8 or 16");
    constexpr int kRing = STAGES / 4;  // ring passes per loader warp
    constexpr int kWtReadyOff = 0;
    constexpr int kActReadyOff = STAGES * 8;
    constexpr int kDataConsumedOff = 2 * STAGES * 8;

    uint64_t tma_param_base;
    asm volatile("mov.b64 %0, %1;" : "=l"(tma_param_base) : "l"((uint64_t)(&tma_params)));

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    __nv_bfloat16* smem_wt = reinterpret_cast<__nv_bfloat16*>(smem_raw + kSmemWtOff);
    const int smem_wt_addr = smem + kSmemWtOff;
    __nv_bfloat16* smem_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + SmemActOff(STAGES));
    const int smem_act_addr = smem + SmemActOff(STAGES);
    float* smem_red = reinterpret_cast<float*>(smem_raw + SmemRedOff(STAGES));
    const int smem_red_addr = smem + SmemRedOff(STAGES);
    __nv_bfloat16* smem_bias = reinterpret_cast<__nv_bfloat16*>(smem_raw + SmemBiasOff(STAGES));
    const int smem_bias_addr = smem + SmemBiasOff(STAGES);
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(tma_param_base + 0)))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(((const void*)(tma_param_base + 128)))) : "memory"); }

    // Mbarrier init: wt_ready / act_ready (STAGES each, one TMA arrival) and
    // data_consumed (STAGES, one arrival per compute lane).
    if (warp == 0) {
        uint32_t leader = elect_sync();
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init_pred(smem + kWtReadyOff + s * 8, 1, leader);
        }
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init_pred(smem + kActReadyOff + s * 8, 1, leader);
        }
        #pragma unroll
        for (int s = 0; s < STAGES; s++) {
            mbarrier_init_pred(smem + kDataConsumedOff + s * 8, 32, leader);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    __syncthreads();

    const int mbar_base = smem;
    const int wt_ready_addr = mbar_base + kWtReadyOff;
    const int act_ready_addr = mbar_base + kActReadyOff;
    const int data_consumed_addr = mbar_base + kDataConsumedOff;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int k_loops_c = (K + 1024 - 1) / 1024;
            int mib_c = blockIdx.x * 16;
            int ni_c = blockIdx.y * 8;
            if (tid < 16) {
                smem_bias[tid] = bias[mib_c + tid];
            }
            float accum[4];
            #pragma unroll
            for (int z = 0; z < 4; z++) {
                accum[z] = 0.0f;
            }
            unsigned int lane_div8 = lane / 8;
            unsigned int lane_mod8 = lane % 8;
            unsigned int row_wt = lane_mod8 + lane_div8 % 2 * 8;
            unsigned int col_off_wt = lane_div8 / 2;
            unsigned int row_act = lane_mod8;
            #pragma unroll 2
            for (unsigned int ki = 0; ki < k_loops_c; ki++) {
                unsigned int stage_c = (unsigned int)warp + 4 * (ki % kRing);
                unsigned int phase_c = ki / kRing & 1;
                mbarrier_wait(wt_ready_addr + (stage_c) * 8, phase_c);
                mbarrier_wait(act_ready_addr + (stage_c) * 8, phase_c);
                #pragma unroll
                for (int su = 0; su < 4; su++) {
                    unsigned int base_wt = smem_wt_addr + (stage_c * 4 + (unsigned int)su) * 2048;
                    unsigned int base_act = smem_act_addr + (stage_c * 4 + (unsigned int)su) * 1024;
                    #pragma unroll
                    for (int kii = 0; kii < 4; kii++) {
                        unsigned int a_frag[4];
                        unsigned int b_frag[2];
                        unsigned int col_w = (unsigned int)(2 * kii) + col_off_wt;
                        unsigned int col_sw_w = row_wt % 8 ^ col_w;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(base_wt + row_wt * 128 + col_sw_w * 16)
                            : "memory");
                        unsigned int col_a = (unsigned int)(2 * kii) + lane_div8;
                        unsigned int col_sw_a = row_act % 8 ^ col_a;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1])
                            : "r"(base_act + row_act * 128 + col_sw_a * 16)
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(data_consumed_addr + (stage_c) * 8);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_red_addr + (unsigned int)(tid * 16)), "r"(float_bits(accum[0])), "r"(float_bits(accum[1])), "r"(float_bits(accum[2])), "r"(float_bits(accum[3])) : "memory");
            asm volatile("barrier.sync 2, 384;" ::: "memory");
            if (warp == 0) {
                float part[12];
                #pragma unroll
                for (int w = 0; w < 3; w++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&part[w * 4])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(w * 4) + 3]))
                        : "r"(smem_red_addr + (unsigned int)((32 + w * 32 + tid) * 16)));
                }
                #pragma unroll
                for (int z_1 = 0; z_1 < 4; z_1++) {
                    accum[z_1] = accum[z_1] + part[z_1] + part[4 + z_1] + part[8 + z_1];
                }
                int tm = mib_c + lane / 4;
                int tn = ni_c + 2 * (lane % 4);
                float bias_lo = smem_bias[lane / 4];
                float bias_hi = smem_bias[lane / 4 + 8];
                float o00 = accum[0] + bias_lo;
                float o01 = accum[1] + bias_lo;
                float o10 = accum[2] + bias_hi;
                float o11 = accum[3] + bias_hi;
                if (tn < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm)) + (0)) = __float2bfloat16_rn(o00);
                    }
                }
                if (tn + 1 < N) {
                    if (tm < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm)) + (0)) = __float2bfloat16_rn(o01);
                    }
                }
                if (tn < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + (tn * M + tm + 8)) + (0)) = __float2bfloat16_rn(o10);
                    }
                }
                if (tn + 1 < N) {
                    if (tm + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(output + ((tn + 1) * M + tm + 8)) + (0)) = __float2bfloat16_rn(o11);
                    }
                }
            }
        }
    // ---- Role: load_wt ----
    } else if (warp >= 4 && warp <= 7) {
        { // load_wt_main
            int k_loops = (K + 1024 - 1) / 1024;
            int mib = blockIdx.x * 16;
            unsigned int wslot = warp % 4;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int ki_1 = 0; ki_1 < k_loops; ki_1++) {
                    unsigned int stage = wslot + 4 * (ki_1 % kRing);
                    unsigned int phase = ki_1 / kRing & 1;
                    int k_base = (ki_1 * 4 + wslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage) * 8, phase ^ 1);
                    mbarrier_arrive_expect_tx(wt_ready_addr + (stage) * 8, 8192);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int dst_wt = smem_wt_addr + (stage * 4 + (unsigned int)i) * 2048;
                        tma_2d_gmem2smem(dst_wt, ((const void*)(tma_param_base + 0)), k_base + i * 64, mib, wt_ready_addr + (stage) * 8);
                    }
                }
                // Drain: wait for the compute warps to release the kRing-1
                // in-flight slots before the final barrier.sync.
                #pragma unroll
                for (int di = 0; di < kRing; di++) {
                    if (di + 1 < kRing) {
                        unsigned int dki = k_loops + di;
                        unsigned int dstage = wslot + 4 * (dki % kRing);
                        unsigned int dphase = dki / kRing & 1;
                        mbarrier_wait(data_consumed_addr + (dstage) * 8, dphase ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    // ---- Role: load_act ----
    } else if (warp >= 8 && warp <= 11) {
        { // load_act_main
            int k_loops_a = (K + 1024 - 1) / 1024;
            int ni = blockIdx.y * 8;
            unsigned int aslot = warp % 4;
            if (elect_sync()) {
                // PDL: only the activation loaders wait for the upstream grid;
                // weight TMA overlaps the preceding kernel's tail, as in the
                // reference `if constexpr (USE_PDL)` block of csrc/tinygemm2.cu.
                if constexpr (USE_PDL) {
                    asm volatile("griddepcontrol.wait;" ::: "memory");
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                }
                #pragma unroll 1
                for (unsigned int ki_2 = 0; ki_2 < k_loops_a; ki_2++) {
                    unsigned int stage_a = aslot + 4 * (ki_2 % kRing);
                    unsigned int phase_a = ki_2 / kRing & 1;
                    int k_base_a = (ki_2 * 4 + aslot) * 256;
                    mbarrier_wait(data_consumed_addr + (stage_a) * 8, phase_a ^ 1);
                    mbarrier_arrive_expect_tx(act_ready_addr + (stage_a) * 8, 4096);
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 4; i_1++) {
                        int dst_act = smem_act_addr + (stage_a * 4 + (unsigned int)i_1) * 1024;
                        tma_2d_gmem2smem(dst_act, ((const void*)(tma_param_base + 128)), k_base_a + i_1 * 64, ni, act_ready_addr + (stage_a) * 8);
                    }
                }
                #pragma unroll
                for (int di_1 = 0; di_1 < kRing; di_1++) {
                    if (di_1 + 1 < kRing) {
                        unsigned int dki_a = k_loops_a + di_1;
                        unsigned int dstage_a = aslot + 4 * (dki_a % kRing);
                        unsigned int dphase_a = dki_a / kRing & 1;
                        mbarrier_wait(data_consumed_addr + (dstage_a) * 8, dphase_a ^ 1);
                    }
                }
            }
            asm volatile("barrier.sync 2, 384;" ::: "memory");
        }
    }
}


// clang-format on

using tvm::ffi::TensorView;

// Measured stage8->stage16 crossover (GB300, CUPTI, flushed-L2 and warm-L2,
// grids of 8/64/128 CTAs): the deep ring's flushed-state gain exceeds its
// ~0.2us warm-state cost from K=4608 upward and grows with K; at K<=4096 the
// two effects cancel or favor stage8. Multi-wave grids stay on stage8 — the
// doubled SMEM footprint halves CTA residency and loses 2-6us there.
constexpr int kStage16MinK = 4608;

// CUDA device ordinals covered by the per-device once-flags below.
constexpr int kMaxDevices = 64;

struct ProblemDims {
  int batch;
  int in_features;
  int out_features;
};

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline std::once_flag& DeviceFlag(std::once_flag (&flags)[kMaxDevices], int device_id) {
  TVM_FFI_ICHECK(device_id >= 0 && device_id < kMaxDevices)
      << "tinygemm2_sm100 supports CUDA device ordinals below " << kMaxDevices << ", got "
      << device_id;
  return flags[device_id];
}

inline int CurrentDevice() {
  int device_id = -1;
  CheckCuda(cudaGetDevice(&device_id), "cudaGetDevice(tinygemm2_sm100)");
  return device_id;
}

// The verdict is a device property; it is evaluated once per device. A failed
// check leaves the flag unset, so the next call on that device fails again.
inline void CheckSm100Family(int device_id) {
  static std::once_flag flags[kMaxDevices];
  std::call_once(DeviceFlag(flags, device_id), [device_id] {
    int major = 0;
    int minor = 0;
    CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
              "cudaDeviceGetAttribute(compute capability major)");
    CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
              "cudaDeviceGetAttribute(compute capability minor)");
#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4))
    constexpr bool kSupportsSm107 = true;
#else
    constexpr bool kSupportsSm107 = false;
#endif
    TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3 || (minor == 7 && kSupportsSm107)))
        << "tinygemm2_sm100 requires an SM100/SM103 (B200/B300) device, or an SM107 (Rubin) "
        << "device with a CUDA 13.4+ toolkit, got sm_" << major << minor;
  });
}

// The SM count is a device constant; read once per device.
inline int NumSms(int device_id) {
  static std::once_flag flags[kMaxDevices];
  static int counts[kMaxDevices];
  std::call_once(DeviceFlag(flags, device_id), [device_id] {
    CheckCuda(cudaDeviceGetAttribute(&counts[device_id], cudaDevAttrMultiProcessorCount, device_id),
              "cudaDeviceGetAttribute(multiprocessor count)");
  });
  return counts[device_id];
}

inline void CheckBf16(const TensorView& t, const char* name) {
  const DLDataType d = t.dtype();
  TVM_FFI_ICHECK(d.code == kDLBfloat && d.bits == 16 && d.lanes == 1)
      << name << " must be bfloat16, got (code=" << int(d.code) << ", bits=" << int(d.bits)
      << ", lanes=" << int(d.lanes) << ")";
}

inline void CheckCudaBf16Contiguous(const TensorView& t, int ndim, const char* name) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(t.ndim() == ndim) << name << " must be " << ndim << "D, got ndim=" << t.ndim();
  TVM_FFI_ICHECK(t.IsContiguous()) << name << " must be contiguous";
  CheckBf16(t, name);
}

// Validate the public `out = input @ weight.T + bias` contract plus the
// kernel's coverage guards: in_features must fit one TMA box; out_features
// must be a positive multiple of the kTileM output tile. The batch axis has
// NO lower guard — the activation descriptor deliberately allows an
// out-of-bounds box on that axis and TMA zero-fills rows past the end, so
// batch 1..7 inputs are valid.
inline ProblemDims CheckInputs(const TensorView& input, const TensorView& weight,
                               const TensorView& bias, const TensorView& out) {
  CheckCudaBf16Contiguous(input, 2, "input");
  CheckCudaBf16Contiguous(weight, 2, "weight");
  CheckCudaBf16Contiguous(bias, 1, "bias");
  CheckCudaBf16Contiguous(out, 2, "out");
  const int device_id = input.device().device_id;
  TVM_FFI_ICHECK(weight.device().device_id == device_id && bias.device().device_id == device_id &&
                 out.device().device_id == device_id)
      << "input/weight/bias/out must live on the same CUDA device";
  CheckSm100Family(device_id);

  const int64_t batch = input.size(0);
  const int64_t in_features = input.size(1);
  const int64_t out_features = weight.size(0);
  TVM_FFI_ICHECK(weight.size(1) == in_features)
      << "weight.shape[1] (" << weight.size(1) << ") must equal input.shape[1] (" << in_features
      << ")";
  TVM_FFI_ICHECK(bias.size(0) == out_features)
      << "bias.shape[0] (" << bias.size(0) << ") must equal weight.shape[0] (" << out_features
      << ")";
  TVM_FFI_ICHECK(out.size(0) == batch && out.size(1) == out_features)
      << "out must have shape (" << batch << ", " << out_features << "), got (" << out.size(0)
      << ", " << out.size(1) << ")";

  TVM_FFI_ICHECK(batch > 0) << "batch must be positive, got " << batch;
  TVM_FFI_ICHECK(in_features >= kTileK)
      << "in_features (" << in_features << ") must be at least " << kTileK << " (one TMA box)";
  TVM_FFI_ICHECK(out_features >= kTileM && out_features % kTileM == 0)
      << "out_features (" << out_features << ") must be a positive multiple of " << kTileM;
  TVM_FFI_ICHECK(batch <= std::numeric_limits<int>::max() &&
                 in_features <= std::numeric_limits<int>::max() &&
                 out_features <= std::numeric_limits<int>::max())
      << "problem dimensions exceed the kernel's i32 scalar range";

  return ProblemDims{static_cast<int>(batch), static_cast<int>(in_features),
                     static_cast<int>(out_features)};
}

// 2D TMA descriptor over a row-major (rows, kTileK-multiple columns) bf16
// matrix: box (kTileK, box_rows), 128B swizzle, no L2 promotion, no OOB fill.
// The weight descriptor (box_rows = kTileM) stays in bounds on both axes
// (CheckInputs guarantees in_features >= kTileK and out_features >= kTileM);
// the activation descriptor (box_rows = kTileN) may exceed the batch for
// batch 1..7 and TMA zero-fills those rows.
inline CUtensorMap EncodeTma(const TensorView& matrix, int box_rows, const char* name) {
  const uint64_t global_dim[2] = {static_cast<uint64_t>(matrix.size(1)),
                                  static_cast<uint64_t>(matrix.size(0))};
  const uint64_t global_strides[1] = {static_cast<uint64_t>(matrix.stride(0)) *
                                      sizeof(__nv_bfloat16)};
  const uint32_t box_dim[2] = {static_cast<uint32_t>(kTileK), static_cast<uint32_t>(box_rows)};
  const uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm;
  const CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, matrix.data_ptr(), global_dim, global_strides,
      box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(r == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for the " << name << " descriptor: CUresult=" << int(r);
  return tm;
}

struct LaunchArgs {
  __nv_bfloat16* out;
  __nv_bfloat16* bias;
  ProblemDims dims;
  TensorMapPack<2> pack;  // maps[0] = weight, maps[1] = activation
  cudaStream_t stream;
  int device_id;
};

// Launch one instantiation. The dynamic-SMEM opt-in is sticky per (kernel,
// device) and is set once; setting it on every launch costs a host API call
// on the critical path of a ~5us kernel. PDL instantiations launch through
// cudaLaunchKernelEx with programmatic stream serialization, matching the
// in-kernel griddepcontrol pair.
template <int STAGES, bool USE_PDL>
void LaunchInstance(const LaunchArgs& args) {
  constexpr int kSmemBytes = SmemBytes(STAGES);
  static std::once_flag flags[kMaxDevices];
  std::call_once(DeviceFlag(flags, args.device_id), [] {
    CheckCuda(cudaFuncSetAttribute(reinterpret_cast<const void*>(&tinygemm2_kernel<STAGES, USE_PDL>),
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
              "cudaFuncSetAttribute(tinygemm2_sm100 dynamic smem)");
  });

  const dim3 grid((args.dims.out_features + kTileM - 1) / kTileM,
                  (args.dims.batch + kTileN - 1) / kTileN);
  const dim3 block(kThreads);

  if constexpr (USE_PDL) {
    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = kSmemBytes;
    config.stream = args.stream;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    CheckCuda(cudaLaunchKernelEx(&config, &tinygemm2_kernel<STAGES, USE_PDL>, args.out, args.bias,
                                 args.dims.out_features, args.dims.batch, args.dims.in_features,
                                 args.pack),
              "cudaLaunchKernelEx(tinygemm2_sm100)");
  } else {
    tinygemm2_kernel<STAGES, USE_PDL><<<grid, block, kSmemBytes, args.stream>>>(
        args.out, args.bias, args.dims.out_features, args.dims.batch, args.dims.in_features,
        args.pack);
    CheckCuda(cudaGetLastError(), "tinygemm2_sm100 kernel launch");
  }
}

template <int STAGES>
void LaunchStages(const LaunchArgs& args, bool use_pdl) {
  if (use_pdl) {
    LaunchInstance<STAGES, true>(args);
  } else {
    LaunchInstance<STAGES, false>(args);
  }
}

// Ring-depth selection, evaluated in the binding like the reference launcher
// selects STAGES. Three tiers:
//   stage4  — K fits one loader iteration, or the grid runs multiple waves
//             (the halved SMEM footprint doubles CTA residency);
//   stage16 — single-wave long-K shapes, where the deep ring hides the
//             elevated cold-miss latency that the 8-deep ring exposes;
//   stage8  — everything between.
inline int SelectStages(const ProblemDims& dims, int num_sms) {
  const int tiles_m = (dims.out_features + kTileM - 1) / kTileM;
  const int tiles_n = (dims.batch + kTileN - 1) / kTileN;
  const int total_ctas = tiles_m * tiles_n;
  if (dims.in_features <= kKPerLoop || total_ctas > 2 * num_sms) return 4;
  if (dims.in_features >= kStage16MinK && total_ctas <= num_sms) return 16;
  return 8;
}

// out = input @ weight.T + bias (bf16, fp32 accumulation), column-major
// epilogue identical to csrc/tinygemm2.cu.
void Run(TensorView input, TensorView weight, TensorView bias, TensorView out, bool use_pdl) {
  static_assert(sizeof(TensorMapBytes) == sizeof(CUtensorMap),
                "kernel tensor-map parameter must be layout-compatible with CUtensorMap");
  LaunchArgs args;
  args.dims = CheckInputs(input, weight, bias, out);
  const CUtensorMap weight_map = EncodeTma(weight, kTileM, "weight");
  const CUtensorMap activation_map = EncodeTma(input, kTileN, "activation");
  std::memcpy(&args.pack.maps[0], &weight_map, sizeof(TensorMapBytes));
  std::memcpy(&args.pack.maps[1], &activation_map, sizeof(TensorMapBytes));
  args.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  args.bias = reinterpret_cast<__nv_bfloat16*>(bias.data_ptr());
  args.stream = get_stream(input.device());
  args.device_id = CurrentDevice();
  switch (SelectStages(args.dims, NumSms(args.device_id))) {
    case 4:
      LaunchStages<4>(args, use_pdl);
      break;
    case 16:
      LaunchStages<16>(args, use_pdl);
      break;
    default:
      LaunchStages<8>(args, use_pdl);
      break;
  }
}

}  // namespace tinygemm2_sm100
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tinygemm2_sm100_op, flashinfer::tinygemm2_sm100::Run);
