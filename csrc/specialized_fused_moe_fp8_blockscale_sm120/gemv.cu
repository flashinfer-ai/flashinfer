// TVM-FFI binding/launch TU for the specialized sm120 fp8 block-scale MoE kernels.
// The framework-independent device code lives in
// include/flashinfer/fused_moe/specialized_fused_moe_fp8_blockscale_sm120/gemv.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/fused_moe/specialized_fused_moe_fp8_blockscale_sm120/gemv.cuh>

namespace ffi = tvm::ffi;

// ---------------------------------------------------------------- workspace
namespace {
constexpr int kMaxDevices = 64;
// All cached state is per device and READ-ONLY (occupancy/cluster probes).
// Mutable scratch is the caller-provided per-call workspace: the Python
// dispatch layer owns one buffer per (device, stream) for eager calls and
// one per CUDA-graph capture, so independently executing calls never share
// live scratch. The workspace's first kWorkspaceDataOffset bytes are the
// shared counter region (zeroed once at allocation by the owner; see the
// layout contract in decode_kernel.cu): the T=1 phase counters own
// [0..256), and every path's data starts at kWorkspaceDataOffset so reuse
// across token counts never overwrites a counter mid-epoch.
constexpr size_t kWorkspaceDataOffset = 2048;
int g_t1_v3_ok[kMaxDevices];
int g_t1_clu[kMaxDevices];
struct GemvCacheInit {
  GemvCacheInit() {
    for (int i = 0; i < kMaxDevices; ++i) g_t1_v3_ok[i] = g_t1_clu[i] = -1;
  }
} g_gemv_cache_init;
inline size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

struct GemvLayout {
  size_t o_pair, o_pos, o_te, o_tp, o_tc, o_nt, o_int, o_of, total;
};

GemvLayout gemv_layout(int T) {
  const int P = T * TOPK;
  const int MT = (T == 1) ? 1 : ((T <= 8) ? 2 : ((T >= 1024) ? 128 : 4));
  long long nt_bound = (P <= NEXP) ? P : NEXP + (long long)(P - NEXP) / MT;
  GemvLayout L{};
  // Data begins past the reserved counter region; the T=1 phase counters
  // live at [0..256) inside it.
  size_t off = kWorkspaceDataOffset;
  L.o_pair = off;
  off = align_up(off + (size_t)P * 4, 256);
  L.o_pos = off;
  off = align_up(off + (size_t)P * 4, 256);
  L.o_te = off;
  off = align_up(off + (size_t)nt_bound * 4, 256);
  L.o_tp = off;
  off = align_up(off + (size_t)nt_bound * 4, 256);
  L.o_tc = off;
  off = align_up(off + (size_t)nt_bound * 4, 256);
  L.o_nt = off;
  off = align_up(off + 4, 256);
  L.o_int = off;
  off = align_up(off + (size_t)P * INT_ * 2, 256);
  L.o_of = off;
  if (T >= 1024) off = align_up(off + (size_t)T * HID * 4, 256);
  L.total = off;
  return L;
}
}  // namespace

size_t GemvWorkspaceBytes(int T) { return gemv_layout(T).total; }

void RunGemv(ffi::TensorView hidden_states, ffi::TensorView w13, ffi::TensorView w13_scale,
             ffi::TensorView w2, ffi::TensorView w2_scale, ffi::TensorView topk_ids,
             ffi::TensorView topk_weights, ffi::TensorView out, cudaStream_t stream,
             void* workspace, size_t workspace_bytes) {
  const int T = static_cast<int>(hidden_states.shape()[0]);
  const int P = T * TOPK;

  const int MT = (T == 1) ? 1 : ((T <= 8) ? 2 : ((T >= 1024) ? 128 : 4));
  long long nt_bound = (P <= NEXP) ? P : NEXP + (long long)(P - NEXP) / MT;

  const GemvLayout L = gemv_layout(T);
  const size_t o_pair = L.o_pair, o_pos = L.o_pos, o_te = L.o_te, o_tp = L.o_tp, o_tc = L.o_tc,
               o_nt = L.o_nt, o_int = L.o_int, o_of = L.o_of;
  const bool BIG = (T >= 1024);
  const int dev = hidden_states.device().device_id;
  TVM_FFI_ICHECK(dev >= 0 && dev < kMaxDevices) << "device index out of range: " << dev;
  TVM_FFI_ICHECK(workspace != nullptr && workspace_bytes >= L.total)
      << "GEMV workspace too small: " << workspace_bytes << " < " << L.total;
  char* base = (char*)workspace;
  int* pair_idx = (int*)(base + o_pair);
  int* pos_of = (int*)(base + o_pos);
  int* tile_e = (int*)(base + o_te);
  int* tile_p0 = (int*)(base + o_tp);
  int* tile_cnt = (int*)(base + o_tc);
  int* ntiles = (int*)(base + o_nt);
  __nv_bfloat16* inter = (__nv_bfloat16*)(base + o_int);
  float* outf = (float*)(base + o_of);

  const int* ids = (const int*)topk_ids.data_ptr();
  const float* tws = (const float*)topk_weights.data_ptr();
  const __nv_bfloat16* x = (const __nv_bfloat16*)hidden_states.data_ptr();
  const uint8_t* pw13 = (const uint8_t*)w13.data_ptr();
  const uint8_t* pw2 = (const uint8_t*)w2.data_ptr();
  const float* pw13s = (const float*)w13_scale.data_ptr();
  const float* pw2s = (const float*)w2_scale.data_ptr();
  __nv_bfloat16* pout = (__nv_bfloat16*)out.data_ptr();

  if (T == 1) {
    if (g_t1_v3_ok[dev] < 0) {
      int coop = 0, sms = 0, occ = 0;
      cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, t1_v3_kernel<T1NR, T1PF>, 256, 0);
      g_t1_v3_ok[dev] = coop && occ * sms >= (2048 / T1NR);
    }
    if (g_t1_v3_ok[dev]) {
      unsigned int* bar = reinterpret_cast<unsigned int*>(base);
      if (g_t1_clu[dev] < 0) {
        int cl = 0;
        cudaDeviceGetAttribute(&cl, cudaDevAttrClusterLaunch, dev);
        int mac = 0;
        if (cl) {
          cudaLaunchConfig_t probe = {};
          probe.gridDim = dim3(2048 / T1NR);
          probe.blockDim = dim3(256);
          cudaLaunchAttribute pa[1];
          pa[0].id = cudaLaunchAttributeClusterDimension;
          pa[0].val.clusterDim.x = T1CLU;
          pa[0].val.clusterDim.y = 1;
          pa[0].val.clusterDim.z = 1;
          probe.attrs = pa;
          probe.numAttrs = 1;
          cudaOccupancyMaxActiveClusters(
              &mac, reinterpret_cast<const void*>(t1_v3_kernel<T1NR, T1PF, true>), &probe);
        }
        g_t1_clu[dev] = (cl && mac >= (2048 / T1NR) / T1CLU) ? 1 : 0;
      }
      if (g_t1_clu[dev]) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(2048 / T1NR);
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cudaLaunchAttribute attrs[3];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = T1CLU;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        attrs[1].id = cudaLaunchAttributeCooperative;
        attrs[1].val.cooperative = 1;
        attrs[2].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
        attrs[2].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
        cfg.attrs = attrs;
        cfg.numAttrs = 3;
        if (cudaLaunchKernelEx(&cfg, t1_v3_kernel<T1NR, T1PF, true>, x, pw13, pw13s, pw2, pw2s, ids,
                               tws, inter, pout, bar) == cudaSuccess)
          return;
        g_t1_clu[dev] = 0;
        (void)cudaGetLastError();  // consume the probe failure before the fallback launch
      }
      // Keep the 512-arrival fallback on its own epoch counter: a prior
      // successful cluster launch advances the cluster counter by only 64.
      unsigned int* flat_bar = bar + 1;
      void* args[] = {&x, &pw13, &pw13s, &pw2, &pw2s, &ids, &tws, &inter, &pout, &flat_bar};
      const cudaError_t st =
          cudaLaunchCooperativeKernel(reinterpret_cast<void*>(t1_v3_kernel<T1NR, T1PF>),
                                      dim3(2048 / T1NR), dim3(256), args, 0, stream);
      TVM_FFI_ICHECK(st == cudaSuccess)
          << "T=1 cooperative launch failed: " << cudaGetErrorString(st);
      return;
    }
    void* args[] = {&x, &pw13, &pw13s, &pw2, &pw2s, &ids, &tws, &inter, &pout};
    const cudaError_t st = cudaLaunchCooperativeKernel(reinterpret_cast<void*>(t1_fused_kernel),
                                                       dim3(512), dim3(256), args, 0, stream);
    // Never return silently with the output uncomputed: on the validated
    // device profile this grid fits; anywhere else the failure must surface.
    TVM_FFI_ICHECK(st == cudaSuccess)
        << "T=1 fused cooperative launch failed: " << cudaGetErrorString(st);
    return;
  }

  prep_kernel<<<1, 256, 0, stream>>>(ids, P, MT, pair_idx, pos_of, tile_e, tile_p0, tile_cnt,
                                     ntiles);
  if (BIG) {
    cudaMemsetAsync(outf, 0, (size_t)T * HID * 4, stream);
    dim3 gg((unsigned)nt_bound, INT_ / 64);
    gemm_fc1<<<gg, 256, 0, stream>>>(x, pw13, pw13s, tile_e, tile_p0, tile_cnt, pair_idx, ntiles,
                                     inter);
    dim3 gh((unsigned)nt_bound, HID / 128);
    gemm_fc2<<<gh, 256, 0, stream>>>(inter, pw2, pw2s, tws, tile_e, tile_p0, tile_cnt, pair_idx,
                                     ntiles, outf);
    long long n = (long long)T * HID;
    int nb = (int)((n + 255) / 256);
    if (nb > 8192) nb = 8192;
    f32_to_bf16<<<nb, 256, 0, stream>>>(outf, pout, n);
    return;
  }
  if (MT == 2) {
    constexpr int NR = 4, RW = 2;
    dim3 g((unsigned)nt_bound, INT_ / (NR * RW));
    fc1_kernel<2, NR, RW, false, false><<<g, 32 * 4 * RW, 0, stream>>>(
        x, pw13, pw13s, tile_e, tile_p0, tile_cnt, pair_idx, ntiles, ids, inter);
  } else if (T == 64) {
    constexpr int NR = 4, RW = 2;
    dim3 g((unsigned)nt_bound, INT_ / (NR * RW));
    fc1_kernel<4, NR, RW, false, true, true><<<g, 32 * 4 * RW, 0, stream>>>(
        x, pw13, pw13s, tile_e, tile_p0, tile_cnt, pair_idx, ntiles, ids, inter);
  } else {
    constexpr int NR = 4, RW = 2;
    dim3 g((unsigned)nt_bound, INT_ / (NR * RW));
    fc1_kernel<4, NR, RW, false, false><<<g, 32 * 4 * RW, 0, stream>>>(
        x, pw13, pw13s, tile_e, tile_p0, tile_cnt, pair_idx, ntiles, ids, inter);
  }
  if (T == 64) {
    dim3 g2((unsigned)T, HID / 8);
    fc2_kernel<8, false, true>
        <<<g2, 32 * TOPK, 0, stream>>>(inter, pw2, pw2s, ids, tws, pos_of, pout);
  } else {
    dim3 g2((unsigned)T, HID / 4);
    fc2_kernel<4, false><<<g2, 32 * TOPK, 0, stream>>>(inter, pw2, pw2s, ids, tws, pos_of, pout);
  }
  const cudaError_t tail_status = cudaGetLastError();
  TVM_FFI_ICHECK(tail_status == cudaSuccess)
      << "GEMV kernel launch failed: " << cudaGetErrorString(tail_status);
}
