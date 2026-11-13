#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace ffi = tvm::ffi;

__device__ __forceinline__ void mma_fp8(float &d0, float &d1, float &d2, float &d3,
                                        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float x, float y) {
  uint32_t out;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(y), "f"(x));
  return out;
}

__device__ __forceinline__ void cp_async_16(void *dst, const void *src) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :: "r"(smem), "l"(src));
}

__device__ __forceinline__ void cp_async_16_keep(void *dst, const void *src) {
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], 16;"
               :: "r"(smem), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group 0;");
}

__device__ __forceinline__ void cp_async_wait1() {
  asm volatile("cp.async.wait_group 1;");
}

__device__ __forceinline__ void cp_async_wait2() {
  asm volatile("cp.async.wait_group 2;");
}

__device__ __forceinline__ void cp_async_wait3() {
  asm volatile("cp.async.wait_group 3;");
}

__device__ __forceinline__ void cp_async_wait4() {
  asm volatile("cp.async.wait_group 4;");
}

template <int CTA_N, int STAGES, int CTA_M = 32>
__global__ __launch_bounds__(CTA_N * CTA_M / 4) void fp8_block_gemm_cta_async(
    const uint8_t *__restrict__ A, const uint8_t *__restrict__ B,
    const float *__restrict__ As, const float *__restrict__ Bs,
    uint16_t *__restrict__ Y) {
  constexpr int M = 32;
  constexpr int N = 12288;
  constexpr int K = 2048;
  constexpr int ROW_BYTES = 144;
  constexpr int THREADS = CTA_N * CTA_M / 4;
  constexpr int A_BYTES = CTA_M * ROW_BYTES;
  constexpr int B_BYTES = CTA_N * ROW_BYTES;
  constexpr int STAGE_BYTES = A_BYTES + B_BYTES;
  __shared__ __align__(16) uint8_t sm[STAGES][STAGE_BYTES];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int group = lane >> 2;
  const int tid4 = lane & 3;
  const int mbase = static_cast<int>(blockIdx.y) * CTA_M;
  int nwarp, m0;
  if constexpr (CTA_M == 16) {
    nwarp = warp;
    m0 = mbase + group;
  } else {
    constexpr int N_WARPS = CTA_N / 8;
    nwarp = warp % N_WARPS;
    m0 = mbase + (warp / N_WARPS) * 16 + group;
  }
  const int m1 = m0 + 8;
  const int mlocal = m0 - mbase;
  constexpr int tile_n = CTA_N;
  const int nbase = static_cast<int>(blockIdx.x) * CTA_N;
  const int nlocal = nwarp * 8 + group;
  constexpr int kb_count = K >> 7;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

  #pragma unroll
  for (int init = 0; init < STAGES - 1; ++init) {
    const int init_k = init << 7;
    #pragma unroll
    for (int bi = tid; bi < tile_n * 8; bi += THREADS) {
      const int row = bi >> 3;
      const int chunk = bi & 7;
      cp_async_16(sm[init] + A_BYTES + row * ROW_BYTES + chunk * 16,
                  B + static_cast<int64_t>(nbase + row) * K + init_k + chunk * 16);
    }
    #pragma unroll
    for (int ai = tid; ai < CTA_M * 8; ai += THREADS) {
      const int row = ai >> 3;
      const int chunk = ai & 7;
      cp_async_16_keep(sm[init] + row * ROW_BYTES + chunk * 16,
                       A + static_cast<int64_t>(mbase + row) * K + init_k +
                           chunk * 16);
    }
    cp_async_commit();
  }
  if constexpr (STAGES == 2) cp_async_wait();
  else if constexpr (STAGES == 6) cp_async_wait4();
  else if constexpr (STAGES == 5) cp_async_wait3();
  else if constexpr (STAGES == 4) cp_async_wait2();
  else cp_async_wait1();
  __syncthreads();

  int buf = 0;
  #pragma unroll 1
  for (int kb = 0; kb < kb_count; ++kb) {
    const bool issue_next = kb + STAGES - 1 < kb_count;
    if (issue_next) {
      const int next_k = (kb + STAGES - 1) << 7;
      const int next_buf = (buf == 0) ? STAGES - 1 : (buf - 1);
      #pragma unroll
      for (int bi = tid; bi < tile_n * 8; bi += THREADS) {
        const int row = bi >> 3;
        const int chunk = bi & 7;
        cp_async_16(sm[next_buf] + A_BYTES + row * ROW_BYTES + chunk * 16,
                    B + static_cast<int64_t>(nbase + row) * K + next_k + chunk * 16);
      }
      #pragma unroll
      for (int ai = tid; ai < CTA_M * 8; ai += THREADS) {
        const int row = ai >> 3;
        const int chunk = ai & 7;
        cp_async_16_keep(sm[next_buf] + row * ROW_BYTES + chunk * 16,
                         A + static_cast<int64_t>(mbase + row) * K + next_k +
                             chunk * 16);
      }
      cp_async_commit();
    }

    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
    #pragma unroll
    for (int kk = 0; kk < 128; kk += 32) {
      const uint8_t *ap0 = sm[buf] + mlocal * ROW_BYTES + kk + (tid4 << 2);
      const uint8_t *ap1 = ap0 + 8 * ROW_BYTES;
      const uint32_t a0 = *reinterpret_cast<const uint32_t *>(ap0);
      const uint32_t a2 = *reinterpret_cast<const uint32_t *>(ap0 + 16);
      const uint32_t a1 = *reinterpret_cast<const uint32_t *>(ap1);
      const uint32_t a3 = *reinterpret_cast<const uint32_t *>(ap1 + 16);
      const uint8_t *bp =
          sm[buf] + A_BYTES + nlocal * ROW_BYTES + kk + (tid4 << 2);
      const uint32_t b0 = *reinterpret_cast<const uint32_t *>(bp);
      const uint32_t b1 = *reinterpret_cast<const uint32_t *>(bp + 16);
      mma_fp8(dot0, dot1, dot2, dot3, a0, a1, a2, a3, b0, b1);
    }

    float bs = 0.0f;
    if (lane == 0) bs = Bs[((nbase + nwarp * 8) >> 7) * kb_count + kb];
    bs = __shfl_sync(0xffffffffu, bs, 0);
    float as0 = 0.0f, as1 = 0.0f;
    if (tid4 == 0) as0 = As[m0 * kb_count + kb];
    if (tid4 == 0) as1 = As[m1 * kb_count + kb];
    as0 = __shfl_sync(0xffffffffu, as0, group << 2);
    as1 = __shfl_sync(0xffffffffu, as1, group << 2);
    const float s0 = as0 * bs;
    const float s1 = as1 * bs;
    acc0 = fmaf(dot0, s0, acc0);
    acc1 = fmaf(dot1, s0, acc1);
    acc2 = fmaf(dot2, s1, acc2);
    acc3 = fmaf(dot3, s1, acc3);

    if (kb + 1 < kb_count) {
      if constexpr (STAGES == 2) {
        cp_async_wait();
      } else if constexpr (STAGES == 6) {
        if (issue_next) cp_async_wait4();
        else if (kb + 4 < kb_count) cp_async_wait3();
        else if (kb + 3 < kb_count) cp_async_wait2();
        else if (kb + 2 < kb_count) cp_async_wait1();
        else cp_async_wait();
      } else if constexpr (STAGES == 5) {
        if (issue_next) cp_async_wait3();
        else if (kb + 3 < kb_count) cp_async_wait2();
        else if (kb + 2 < kb_count) cp_async_wait1();
        else cp_async_wait();
      } else if constexpr (STAGES == 4) {
        if (issue_next) cp_async_wait2();
        else if (kb + 2 < kb_count) cp_async_wait1();
        else cp_async_wait();
      } else {
        if (issue_next) cp_async_wait1();
        else cp_async_wait();
      }
      __syncthreads();
      buf = (buf == STAGES - 1) ? 0 : (buf + 1);
    }
  }

  const int nout = nbase + nwarp * 8 + (tid4 << 1);
  *reinterpret_cast<uint32_t *>(Y + static_cast<int64_t>(m0) * N + nout) =
      pack_bf16(acc0, acc1);
  *reinterpret_cast<uint32_t *>(Y + static_cast<int64_t>(m1) * N + nout) =
      pack_bf16(acc2, acc3);
}

void LaunchAsyncM32N12288(const uint8_t *a, const uint8_t *b,
                          const float *as, const float *bs,
                          __nv_bfloat16 *y, int M, int N, int K,
                          cudaStream_t stream) {
  dim3 grid(N / 96, 1);
  fp8_block_gemm_cta_async<96, 3><<<grid, 768, 0, stream>>>(
      a, b, as, bs, reinterpret_cast<uint16_t *>(y));
}

// M16xN2048xK4096 specialization.  The direct-fragment kernel's NCU trace is
// dominated by long-scoreboard waits on the streamed B loads.  Keep A in the
// read-only path, but cooperatively double-buffer the next eight 128-wide B
// groups through shared memory while the current eight groups execute.
struct AsyncFrag8 { uint32_t v[8]; };

__device__ __forceinline__ AsyncFrag8 async_ld_keep8(const void *p) {
  AsyncFrag8 r;
  asm("ld.global.nc.L1::evict_last.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]),
        "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}

__device__ __forceinline__ AsyncFrag8 async_ld_shared8(const void *p) {
  AsyncFrag8 r;
  const uint4 x0 = *reinterpret_cast<const uint4 *>(p);
  const uint4 x1 = *reinterpret_cast<const uint4 *>(
      static_cast<const uint8_t *>(p) + 16);
  r.v[0] = x0.x; r.v[1] = x0.y; r.v[2] = x0.z; r.v[3] = x0.w;
  r.v[4] = x1.x; r.v[5] = x1.y; r.v[6] = x1.z; r.v[7] = x1.w;
  return r;
}

__device__ __forceinline__ float async_ld_keepf(const float *p) {
  float r;
  asm("ld.global.nc.L1::evict_last.f32 %0, [%1];" : "=f"(r) : "l"(p));
  return r;
}

__global__ __launch_bounds__(512, 1) void fp8_m16n2048k4096_async_b(
    const uint8_t *__restrict__ A, const uint8_t *__restrict__ B,
    const float *__restrict__ As, const float *__restrict__ Bs,
    __nv_bfloat16 *__restrict__ Y) {
  constexpr int M = 16, N = 2048, K = 4096, KB = 32;
  constexpr int CTA_N = 32, NW_N = 2, NW_K = 8, NT = 2;
  constexpr int THREADS = 512, STEP_K = NW_K * 128;
  // A 16-byte pad rotates successive rows by four shared-memory banks.  With
  // the natural 1024-byte stride all eight lane groups alias the same banks.
  constexpr int B_ROW_BYTES = STEP_K + 16;
  constexpr int B_STAGE_BYTES = CTA_N * B_ROW_BYTES;
  constexpr int B_PIPE_BYTES = 2 * B_STAGE_BYTES;
  constexpr int NACC = NT * 4;
  constexpr int RED_FLOATS = NW_N * NW_K * NACC * 32;
  extern __shared__ __align__(16) uint8_t storage[];
  uint8_t *bpipe = storage;
  float *red = reinterpret_cast<float *>(storage + B_PIPE_BYTES);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int ng = warp & 1;
  const int wk = warp >> 1;
  const int g = lane >> 2;
  const int c = lane & 3;
  const int nbase = static_cast<int>(blockIdx.x) * CTA_N;

  auto issue_b_stage = [&](int stage, int step) {
#pragma unroll
    for (int chunk = tid; chunk < CTA_N * STEP_K / 16; chunk += THREADS) {
      const int row = chunk >> 6;       // 64 sixteen-byte chunks per 1024-B row
      const int part = chunk & 63;
      cp_async_16(bpipe + stage * B_STAGE_BYTES + row * B_ROW_BYTES + part * 16,
                  B + static_cast<int64_t>(nbase + row) * K +
                      step * STEP_K + part * 16);
    }
    cp_async_commit();
  };

  issue_b_stage(0, 0);
  issue_b_stage(1, 1);
  // Drain the prologue fully.  Later iterations still overlap stage 2/3 with
  // compute; this also makes both parity buffers visible before first use.
  cp_async_wait();
  __syncthreads();

  float acc[NACC] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    const int gk = step * NW_K + wk;
    const uint8_t *ap0 = A + static_cast<int64_t>(g) * K + gk * 128 + c * 32;
    const uint8_t *ap1 = ap0 + 8 * K;
    const AsyncFrag8 af0 = async_ld_keep8(ap0);
    const AsyncFrag8 af1 = async_ld_keep8(ap1);

    float as0 = (c == 0) ? async_ld_keepf(As + g * KB + gk) : 0.f;
    float as1 = (c == 0) ? async_ld_keepf(As + (g + 8) * KB + gk) : 0.f;
    as0 = __shfl_sync(0xffffffffu, as0, g * 4);
    as1 = __shfl_sync(0xffffffffu, as1, g * 4);
    float bsv = (lane == 0) ? async_ld_keepf(Bs + (nbase >> 7) * KB + gk) : 0.f;
    bsv = __shfl_sync(0xffffffffu, bsv, 0);
    const float s0 = as0 * bsv;
    const float s1 = as1 * bsv;

#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int brow = ng * 16 + i * 8 + g;
      const uint8_t *bp = bpipe + (step & 1) * B_STAGE_BYTES +
                          brow * B_ROW_BYTES + wk * 128 + c * 32;
      const AsyncFrag8 bf = async_ld_shared8(bp);
      float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        mma_fp8(d0, d1, d2, d3, af0.v[2 * j], af1.v[2 * j],
                af0.v[2 * j + 1], af1.v[2 * j + 1],
                bf.v[2 * j], bf.v[2 * j + 1]);
      acc[i * 4 + 0] = fmaf(d0, s0, acc[i * 4 + 0]);
      acc[i * 4 + 1] = fmaf(d1, s0, acc[i * 4 + 1]);
      acc[i * 4 + 2] = fmaf(d2, s1, acc[i * 4 + 2]);
      acc[i * 4 + 3] = fmaf(d3, s1, acc[i * 4 + 3]);
    }

    if (step + 1 < 4) {
      __syncthreads();
      if (step + 2 < 4) issue_b_stage(step & 1, step + 2);
      if (step + 2 < 4) cp_async_wait1();
      else cp_async_wait();
      __syncthreads();
    }
  }

  float *sw = red + static_cast<size_t>(warp) * NACC * 32;
#pragma unroll
  for (int a = 0; a < NACC; ++a) sw[a * 32 + lane] = acc[a];
  __syncthreads();

  // One output element per thread; sum the eight K-warps for its N group.
  const int idx = tid;
  const int ngi = idx >> 8;
  const int rest = idx & 255;
  const int a = rest >> 5;
  const int l = rest & 31;
  float sum = 0.f;
#pragma unroll
  for (int w = 0; w < NW_K; ++w)
    sum += red[((w * NW_N + ngi) * NACC + a) * 32 + l];
  const int i = a >> 2;
  const int sl = a & 3;
  const int row = ((sl >= 2) ? 8 : 0) + (l >> 2);
  const int col = nbase + ngi * 16 + i * 8 + (l & 3) * 2 + (sl & 1);
  Y[static_cast<int64_t>(row) * N + col] = __float2bfloat16_rn(sum);
}

void RunAsyncDeep(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As,
                  ffi::TensorView Bs, ffi::TensorView Y) {
  constexpr int SMEM = 2 * 32 * (1024 + 16) +
                       2 * 8 * 8 * 32 * sizeof(float);
  const auto *ap = static_cast<const uint8_t *>(A.data_ptr());
  const auto *bp = static_cast<const uint8_t *>(B.data_ptr());
  const auto *asp = static_cast<const float *>(As.data_ptr());
  const auto *bsp = static_cast<const float *>(Bs.data_ptr());
  auto *yp = static_cast<__nv_bfloat16 *>(Y.data_ptr());
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(kDLCUDA, A.device().device_id));
  cudaFuncSetAttribute(fp8_m16n2048k4096_async_b,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  fp8_m16n2048k4096_async_b<<<64, 512, SMEM, stream>>>(ap, bp, asp, bsp, yp);
}





