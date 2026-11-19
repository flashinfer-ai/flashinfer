// TVM-FFI binding/launch TU for the specialized sm120 fp8 block-scale MoE kernels.
// The framework-independent device code lives in
// include/flashinfer/fused_moe/specialized_fused_moe_fp8_blockscale_sm120/decode_kernel.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/fused_moe/specialized_fused_moe_fp8_blockscale_sm120/decode_kernel.cuh>

namespace ffi = tvm::ffi;

void RunOurs(ffi::TensorView hidden_states, ffi::TensorView w13, ffi::TensorView w13_scale,
             ffi::TensorView w2, ffi::TensorView w2_scale, ffi::TensorView topk_ids,
             ffi::TensorView topk_weights, ffi::TensorView out, void* workspace,
             size_t workspace_bytes);
void RunOursFlat(ffi::TensorView hidden_states, ffi::TensorView w13, ffi::TensorView w13_scale,
                 ffi::TensorView w2, ffi::TensorView w2_scale, ffi::TensorView topk_ids,
                 ffi::TensorView topk_weights, ffi::TensorView out, void* workspace,
                 size_t workspace_bytes);
void RunGemv(ffi::TensorView hidden_states, ffi::TensorView w13, ffi::TensorView w13_scale,
             ffi::TensorView w2, ffi::TensorView w2_scale, ffi::TensorView topk_ids,
             ffi::TensorView topk_weights, ffi::TensorView out, cudaStream_t stream,
             void* workspace, size_t workspace_bytes);
size_t GemvWorkspaceBytes(int T);
size_t PrefillWorkspaceBytes(int tokens);
size_t PrefillFlatWorkspaceBytes(int tokens);

// ---------------------------------------------------------------- host

// All cached state is per device, and READ-ONLY device answers only (device
// probes, shared-memory opt-ins): no mutable scratch is cached at module
// scope. Mutable scratch is the caller-provided per-call workspace, owned by
// the Python dispatch layer per (device, stream) for eager calls and per
// CUDA-graph capture for graphs, so independently executing calls and
// independently captured graphs never alias live scratch.
static constexpr int kMaxDevices = 64;

static int g_persistent_sms[kMaxDevices] = {};
static int g_persistent_13_ok = -1;
static int g_t4_pair_cluster_ok[kMaxDevices];
static struct DecodeCacheInit {
  DecodeCacheInit() {
    for (int i = 0; i < kMaxDevices; ++i) g_t4_pair_cluster_ok[i] = -1;
  }
} g_decode_cache_init;

// Workspace bytes the dispatch for T tokens requires. The Python layer sizes
// the per-(device, stream) / per-capture workspace it passes to Run with
// this; keep the two dispatch trees in exact agreement.
int64_t PlanWorkspaceBytes(int64_t T) {
  if (T == 1) return (int64_t)GemvWorkspaceBytes((int)T);
  if (T <= 16) return (int64_t)kPersistentDecodeWsBytes;
  if (T < 128) return (int64_t)GemvWorkspaceBytes((int)T);
  if (T < 16384) return (int64_t)PrefillWorkspaceBytes((int)T);
  return (int64_t)PrefillFlatWorkspaceBytes((int)T);
}

void Run(ffi::TensorView hidden_states, ffi::TensorView w13, ffi::TensorView w13_scale,
         ffi::TensorView w2, ffi::TensorView w2_scale, ffi::TensorView topk_ids,
         ffi::TensorView topk_weights, ffi::TensorView out, ffi::TensorView workspace) {
  const int dev = hidden_states.device().device_id;
  TVM_FFI_ICHECK(dev >= 0 && dev < kMaxDevices) << "device index out of range: " << dev;
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, dev));
  const int T = (int)hidden_states.shape()[0];
  TVM_FFI_ICHECK(workspace.ndim() == 1) << "workspace must be a 1-D byte tensor";
  TVM_FFI_ICHECK(workspace.device().device_id == dev)
      << "workspace must live on the input's device";
  const size_t ws_bytes = static_cast<size_t>(workspace.shape()[0]) * (workspace.dtype().bits / 8);
  void* ws = workspace.data_ptr();
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(ws) % 256 == 0)
      << "workspace must be 256-byte aligned";
  TVM_FFI_ICHECK(ws_bytes >= static_cast<size_t>(PlanWorkspaceBytes(T)))
      << "workspace too small: " << ws_bytes << " < " << PlanWorkspaceBytes(T)
      << " bytes for T=" << T;
  if (T == 1) {
    RunGemv(hidden_states, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights, out, stream, ws,
            ws_bytes);
    return;
  }
  if (T >= 128) {
    if (T >= 16384)
      RunOursFlat(hidden_states, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights, out, ws,
                  ws_bytes);
    else
      RunOurs(hidden_states, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights, out, ws,
              ws_bytes);
    return;
  }
  const uint8_t* pw13 = (const uint8_t*)w13.data_ptr();
  const uint8_t* pw2 = (const uint8_t*)w2.data_ptr();
  const float* ps13 = (const float*)w13_scale.data_ptr();
  const float* ps2 = (const float*)w2_scale.data_ptr();
  const int* pids = (const int*)topk_ids.data_ptr();
  const float* ptw = (const float*)topk_weights.data_ptr();
  __nv_bfloat16* pout = (__nv_bfloat16*)out.data_ptr();

  if (T <= 16) {
    if (g_persistent_sms[dev] == 0)
      cudaDeviceGetAttribute(&g_persistent_sms[dev], cudaDevAttrMultiProcessorCount, dev);
    if (g_persistent_13_ok < 0) {
      int occ = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_persistent_decode<13, true>, 416, 0);
      g_persistent_13_ok = occ >= 2;
    }
    const bool use_two = T == 4 && g_persistent_13_ok;
    const int blocks = use_two ? 2 * g_persistent_sms[dev] : g_persistent_sms[dev];
    const int work_blocks = use_two ? blocks / T : blocks;
    const int w1 = (use_two ? TOPK : T * TOPK) * (I_DIM / 2);
    const int w2n = (use_two ? 1 : T) * (H_DIM / 2);
    int q1 = w1 / work_blocks, r1 = w1 % work_blocks;
    int q2 = w2n / work_blocks, r2 = w2n % work_blocks;
    // Per-call barrier state and intermediate, carved from the workspace.
    // The counter region was zeroed when the workspace was allocated (see
    // the layout contract above); the 64-bit epoch counters then stay
    // aligned across stream-ordered launches without per-launch resets.
    unsigned long long* bars = reinterpret_cast<unsigned long long*>(ws);
    __nv_bfloat16* inter =
        reinterpret_cast<__nv_bfloat16*>(static_cast<uint8_t*>(ws) + kWorkspaceDataOffset);
    const __nv_bfloat16* px = (const __nv_bfloat16*)hidden_states.data_ptr();
    int targ = T;
    // The software grid barriers wait for every CTA of the grid, so each of
    // these launches must guarantee co-residency. __launch_bounds__ only
    // proves each CTA CAN fit; cudaLaunchCooperativeKernel (or
    // cudaLaunchKernelEx with the cooperative attribute) makes the driver
    // admit the grid only when all CTAs are simultaneously resident, and
    // fails the launch otherwise. A failed cooperative launch fails CLOSED
    // through TVM_FFI_ICHECK: there is no plain-launch fallback, because a
    // plain launch of a spin-barrier grid without residency can deadlock.
    if (use_two) {
      if (g_t4_pair_cluster_ok[dev] < 0) {
        int cluster_launch = 0, max_active_clusters = 0;
        cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, dev);
        if (cluster_launch && g_persistent_sms[dev] == 188 && blocks == 376) {
          cudaLaunchConfig_t probe = {};
          probe.gridDim = dim3(blocks);
          probe.blockDim = dim3(416);
          cudaLaunchAttribute probe_attr[1];
          probe_attr[0].id = cudaLaunchAttributeClusterDimension;
          probe_attr[0].val.clusterDim.x = 2;
          probe_attr[0].val.clusterDim.y = 1;
          probe_attr[0].val.clusterDim.z = 1;
          probe.attrs = probe_attr;
          probe.numAttrs = 1;
          cudaOccupancyMaxActiveClusters(
              &max_active_clusters,
              reinterpret_cast<const void*>(k_persistent_decode<13, true, true>), &probe);
        }
        g_t4_pair_cluster_ok[dev] = cluster_launch && max_active_clusters >= 188 ? 1 : 0;
      }
      if (g_t4_pair_cluster_ok[dev]) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(blocks);
        cfg.blockDim = dim3(416);
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cudaLaunchAttribute attrs[2];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = 2;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        attrs[1].id = cudaLaunchAttributeCooperative;
        attrs[1].val.cooperative = 1;
        cfg.attrs = attrs;
        cfg.numAttrs = 2;
        if (cudaLaunchKernelEx(&cfg, k_persistent_decode<13, true, true>, px, pw13, ps13, pw2, ps2,
                               pids, ptw, pout, T, q1, r1, q2, r2, bars, inter) == cudaSuccess)
          return;
        g_t4_pair_cluster_ok[dev] = 0;
        (void)cudaGetLastError();  // consume the probe failure before the fallback launch
      }
      // Cooperative (non-cluster) fallback: still a residency-guaranteed
      // launch of the same barrier protocol. If even this cannot be
      // admitted (e.g. under MPS with a reduced SM share), fail closed.
      void* t4_args[] = {&px,   &pw13, &ps13, &pw2, &ps2, &pids, &ptw,  &pout,
                         &targ, &q1,   &r1,   &q2,  &r2,  &bars, &inter};
      const cudaError_t t4_status =
          cudaLaunchCooperativeKernel(reinterpret_cast<void*>(k_persistent_decode<13, true>),
                                      dim3(blocks), dim3(416), t4_args, 0, stream);
      TVM_FFI_ICHECK(t4_status == cudaSuccess)
          << "persistent decode T=4 cooperative launch failed (no residency guarantee; refusing "
             "the unsafe plain launch): "
          << cudaGetErrorString(t4_status);
    } else if (T == 16) {
      void* t16_args[] = {&px,   &pw13, &ps13, &pw2, &ps2, &pids, &ptw,  &pout,
                          &targ, &q1,   &r1,   &q2,  &r2,  &bars, &inter};
      const cudaError_t t16_status =
          cudaLaunchCooperativeKernel(reinterpret_cast<void*>(k_persistent_decode<22, 2>),
                                      dim3(blocks), dim3(704), t16_args, 0, stream);
      TVM_FFI_ICHECK(t16_status == cudaSuccess)
          << "persistent decode T=16 cooperative launch failed (no residency guarantee; refusing "
             "the unsafe plain launch): "
          << cudaGetErrorString(t16_status);
    } else {
      void* gen_args[] = {&px,   &pw13, &ps13, &pw2, &ps2, &pids, &ptw,  &pout,
                          &targ, &q1,   &r1,   &q2,  &r2,  &bars, &inter};
      const cudaError_t gen_status =
          cudaLaunchCooperativeKernel(reinterpret_cast<void*>(k_persistent_decode<22>),
                                      dim3(blocks), dim3(704), gen_args, 0, stream);
      TVM_FFI_ICHECK(gen_status == cudaSuccess)
          << "persistent decode cooperative launch failed (no residency guarantee; refusing the "
             "unsafe plain launch): "
          << cudaGetErrorString(gen_status);
    }
    return;
  }
  RunGemv(hidden_states, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights, out, stream, ws,
          ws_bytes);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_workspace_bytes, PlanWorkspaceBytes);
