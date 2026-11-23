// Bounded parallel recovery. Included after the ordered recurrence helpers.
#pragma once
namespace tile_recovery {
struct Args {
  const bf16 *x, *dt;
  const float* A;
  const bf16 *B, *C, *D, *bias;
  bf16* y;
  int T, nh;
};
constexpr int cap = 4, W = 4;
constexpr int64_t limit = 64LL << 20;
struct Head {
  int status, count, actual, version, ids[cap];
};
using Arena = unsigned char;
struct Layout {
  int64_t K, T, off[16], used;
};
// Scratch offsets: 10 Meta[slot,T], 11 float2 M,N,
// 12 float2 exclusive affine prefix, 13 Seg[slot,ceil(T/32),32],
// 14 float2 E_after,R, 15 Vote[slot], 16 total USED. Row-column tables are [slot,T,32].
// off: directory, double coeff[slot,T,2], float alpha[slot,K],
// Delta, seed, end (each float[slot,K,32,128]), raw float[slot,T,32], end offset.
static bool try_layout(int64_t T, int64_t nh, Layout& result) noexcept {
  if (T <= 0 || T > INT_MAX - 127 || (nh != 64 && nh != 128)) return false;
  Layout l{};
  l.T = T;
  l.K = (T + 63) / 64;
  // Products fit signed 64 bits even at INT_MAX and cap=4. Guard before allocation/launch.
  int64_t state = cap * l.K * 32 * 128 * 4;
  int64_t sizes[] = {int64_t(sizeof(Head)), cap * T * 16, cap * l.K * 4, state, state, state,
                     cap * T * 32 * 4};
  for (int i = 0; i < 7; ++i) {
    l.off[i] = l.used;
    int64_t n = (sizes[i] + 15) & ~int64_t(15);
    if (n > limit - l.used) return false;
    l.used += n;
  }
  l.off[7] = l.used;
  int64_t cells = cap * T * 32;
  int64_t extra[] = {cap * T * 32, cells * 8, cells * 8, cap * ((T + 31) / 32) * 32 * 16,
                     cells * 8,    cap * 24};
  for (int i = 10; i <= 15; i++) {
    l.off[i] = l.used;
    int64_t n = (extra[i - 10] + 15) & ~int64_t(15);
    if (n > limit - l.used) return false;
    l.used += n;
  }
  result = l;
  return true;
}
template <class U>
__device__ __forceinline__ U* ptr(Arena* w, Layout l, int i) {
  return reinterpret_cast<U*>(w + l.off[i]);
}
__global__ void directory(const int* routes, int nh, Arena* w, Layout l) {
  for (int j = 0; j < 6 * cap; j++) ptr<unsigned>(w, l, 15)[j] = 0;
  Head h{};
  h.version = 2;
  for (int j = 0; j < cap; j++) h.ids[j] = -1;
  for (int i = 0; i < 2 * nh; i++) {
    int r = routes[i];
    if (r == 8 || (nh == 128 && r == 16))
      ++h.actual;
    else if (r)
      h.status |= 1;
  }
  if (h.actual > cap) h.status |= 2;
  if (!h.status)
    for (int i = 0; i < 2 * nh; i++)
      if (routes[i]) h.ids[h.count++] = i;
  *reinterpret_cast<Head*>(w) = h;
}
__device__ __forceinline__ float f(bf16 x) { return __bfloat162float(x); }
__global__ void coefficients(Args v, Arena* w, Layout l) {
  int s = blockIdx.y, k = blockIdx.x, j = threadIdx.x;
  Head* hdr = ptr<Head>(w, l, 0);
  if (s >= hdr->count) return;
  int h = hdr->ids[s] / 2;
  int64_t t = int64_t(k) * 64 + j;
  __shared__ float df[64];
  if (t < l.T) {
    double z = __dadd_rn(double(f(v.dt[t * v.nh + h])), double(f(v.bias[h])));
    double a = z > 20. ? z : log1p(exp(z));
    double d = exp(__dmul_rn(a, double(v.A[h])));
    double* q = ptr<double>(w, l, 1) + 2 * (s * l.T + t);
    q[0] = a;
    q[1] = d;
    df[j] = __double2float_rn(d);
  }
  __syncthreads();
  if (!j) {
    float a = 1.f;
    for (int i = 0; i < 64 && int64_t(k) * 64 + i < l.T; i++) a = __fmul_rn(a, df[i]);
    ptr<float>(w, l, 2)[s * l.K + k] = a;
  }
}
__global__ void prefix(Arena* w, Layout l) {
  int s = blockIdx.y;
  if (s >= ptr<Head>(w, l, 0)->count) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float c = 0.f;
  for (int64_t k = 0; k < l.K; k++) {
    int64_t tile = s * l.K + k, u = tile * 4096 + i;
    ptr<float>(w, l, 4)[u] = c;
    c = __fadd_rn(__fmul_rn(c, ptr<float>(w, l, 2)[tile]), ptr<float>(w, l, 3)[u]);
  }
}
template <bool emit>
__global__ void tiles(Args v, Arena* w, Layout l) {
  int k = blockIdx.x, s = blockIdx.z, tid = threadIdx.x, lane = tid % 32,
      p = blockIdx.y * W + tid / 32;
  Head* hdr = ptr<Head>(w, l, 0);
  if (s >= hdr->count) return;
  int id = hdr->ids[s], h = id / 2, g = h / (v.nh / 8);
  int64_t base = int64_t(k) * 64, tile = s * l.K + k;
  int rows = int(l.T - base < 64 ? l.T - base : 64);
  __shared__ bf16 bb[64][128], cc[emit ? 64 : 1][128];
  __shared__ float sf[64], df[64], xx[64][W];
  for (int i = tid; i < rows * 128; i += 32 * W) {
    int j = i / 128, n = i % 128;
    int64_t u = ((base + j) * 8 + g) * 128 + n;
    bb[j][n] = v.B[u];
    if (emit) cc[j][n] = v.C[u];
  }
  if (tid < rows) {
    const double* q = ptr<double>(w, l, 1) + 2 * (s * l.T + base + tid);
    sf[tid] = __double2float_rn(q[0]);
    df[tid] = __double2float_rn(q[1]);
  }
  for (int i = tid; i < rows * W; i += 32 * W) {
    int j = i / W, n = i % W;
    xx[j][n] = f(v.x[((base + j) * v.nh + h) * 64 + 32 * (id % 2) + blockIdx.y * W + n]);
  }
  __syncthreads();
  int64_t u = (tile * 32 + p) * 128 + lane * 4;
  float st[4];
#pragma unroll
  for (int n = 0; n < 4; n++) st[n] = emit ? ptr<float>(w, l, 4)[u + n] : 0.f;
  float D = emit ? f(v.D[h]) : 0.f;
  int fault = 0;
  float M = 0;
  if (emit) {
    for (int n = 0; n < 4; n++) {
      float a = st[n];
      if (!(fabsf(a) < 0x1p100f)) fault = 4;
      M = fmaxf(M, fabsf(a));
    }
    M = __uint_as_float(__reduce_max_sync(~0u, __float_as_uint(M)));
  }
  for (int j = 0; j < rows; j++) {
    float x = xx[j][tid / 32], sx = __fmul_rn(sf[j], x), a[4];
#pragma unroll
    for (int n = 0; n < 4; n++) {
      float inj = __fmul_rn(sx, f(bb[j][lane * 4 + n]));
      float old = st[n];
      st[n] = old == 0.f ? inj : __fadd_rn(__fmul_rn(old, df[j]), inj);
      if (emit) a[n] = __fmul_rn(st[n], f(cc[j][lane * 4 + n]));
    }
    if (emit) {
      float N = 0;
      for (int n = 0; n < 4; n++) {
        if (!(fabsf(st[n]) < 0x1p100f)) fault = 4;
        N = fmaxf(N, fabsf(st[n]));
      }
      N = __uint_as_float(__reduce_max_sync(~0u, __float_as_uint(N)));
      if (!lane) ptr<float2>(w, l, 11)[(s * l.T + base + j) * 32 + p] = make_float2(M, N);
      M = N;
      float sum = __fadd_rn(__fadd_rn(a[0], a[1]), __fadd_rn(a[2], a[3]));
#pragma unroll
      for (int d = 1; d <= 16; d *= 2) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, d));
      if (!lane) {
        int64_t t = base + j;
        float y = __fadd_rn(sum, __fmul_rn(D, x));
        ptr<float>(w, l, 6)[(s * l.T + t) * 32 + p] = y;
        v.y[(t * v.nh + h) * 64 + 32 * (id % 2) + p] = __float2bfloat16_rn(y);
      }
    }
  }
#pragma unroll
  for (int n = 0; n < 4; n++) ptr<float>(w, l, emit ? 5 : 3)[u + n] = st[n];
  if (emit) {
    fault = __reduce_or_sync(~0u, fault);
    if (!lane && fault) atomicOr(ptr<int>(w, l, 15) + 6 * s + 4, fault);
  }
}
#include "ssd_tile_certificate.cuh"
static_assert(sizeof(Meta) == 32 && sizeof(Seg) == 16 && sizeof(Vote) == 24 &&
              offsetof(Vote, fault) == 16 && sizeof(float2) == 8);
__global__ void joins(Arena* w, Layout l) {
  int s = blockIdx.z, c = blockIdx.y * 8 + threadIdx.x / 32, i = threadIdx.x % 32, k = blockIdx.x;
  if (s >= ptr<Head>(w, l, 0)->count) return;
  float m = 0;
  int bad = 0;
  int64_t u = ((s * l.K + k) * 32 + c) * 128 + i * 4;
  for (int n = 0; n < 4; n++) {
    float a = ptr<float>(w, l, 4)[u + n], b = k ? ptr<float>(w, l, 5)[u + n - 4096] : 0.f;
    float d = fmaxf(__fsub_ru(a, b), __fsub_ru(b, a));
    if (!(d >= 0 && d < 0x1p100f)) bad = 4;
    m = fmaxf(m, d);
  }
  m = wmax(m);
  bad = __reduce_or_sync(~0u, bad);
  if (!i) {
    ptr<Seg>(w, l, 13)[(s * ((l.T + 31) / 32) + 2 * k) * 32 + c].jump = m;
    if (bad) atomicOr(&ptr<Vote>(w, l, 15)[s].fault, bad);
  }
}
static void prepare() {
  TVM_FFI_CHECK_CUDA_ERROR(cudaFuncSetAttribute(
      replay8, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(win1306::Scratch<8>)));
}
static void submit(Args v, const int* routes, Arena* w, Layout l, cudaStream_t stream) {
  dim3 grid(unsigned(l.K), 32 / W, cap);
  directory<<<1, 1, 0, stream>>>(routes, v.nh, w, l);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  coefficients<<<dim3(unsigned(l.K), cap), 64, 0, stream>>>(v, w, l);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  tiles<false><<<grid, W * 32, 0, stream>>>(v, w, l);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  prefix<<<dim3(32, cap), 128, 0, stream>>>(w, l);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  tiles<true><<<grid, W * 32, 0, stream>>>(v, w, l);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  stats<<<dim3(unsigned((l.T + 7) / 8), cap), 256, 0, stream>>>(v, w, l);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  joins<<<dim3(unsigned(l.K), 4, cap), 256, 0, stream>>>(w, l);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  dim3 cg(unsigned((l.T + 31) / 32), 4, cap);
  segments<false><<<cg, 256, 0, stream>>>(v, w, l);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  carries<<<cap, 32, 0, stream>>>(w, l);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  segments<true><<<cg, 256, 0, stream>>>(v, w, l);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  decide<<<1, cap, 0, stream>>>(w, l);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
  replay8<<<dim3(1, 2 * v.nh), 256, sizeof(win1306::Scratch<8>), stream>>>(v, w, l);
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetLastError());
}
}  // namespace tile_recovery
