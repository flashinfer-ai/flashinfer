// TVM-FFI binding/launch TU for the SM90 W4A16 mxfp4 latent routed MoE kernels.
// The framework-independent device code lives in
// include/flashinfer/fused_moe/fused_moe_w4a16_mxfp4_latent_sm90/kernels.cuh;
// this TU keeps the host-side launch glue and dispatch.

#include <atomic>
#include <flashinfer/fused_moe/fused_moe_w4a16_mxfp4_latent_sm90/kernels.cuh>
#include <mutex>

#include "tvm_ffi_utils.h"

struct WorkspaceSlot {
  bool used = false;
  int device = -1;
  cudaStream_t stream = nullptr;
  void* ptr = nullptr;
  size_t size = 0;
  std::mutex enqueue_mu;
};
static WorkspaceSlot g_ws_slots[64];
static std::mutex g_ws_registry_mu;
static bool g_attr[64] = {};

namespace {

inline void check_cuda(cudaError_t status, const char* what) {
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "w4a16_mxfp4_latent_routed_moe: " << what << " failed: " << cudaGetErrorString(status);
}

// Makes x's device current for the whole enqueue sequence (allocations,
// occupancy queries and kernel launches all act on the CURRENT device) and
// restores the caller's device on scope exit, including on exceptions.
class ScopedSetDevice {
 public:
  explicit ScopedSetDevice(int device) {
    check_cuda(cudaGetDevice(&previous_), "cudaGetDevice");
    if (previous_ != device) {
      check_cuda(cudaSetDevice(device), "cudaSetDevice");
      restore_ = true;
    }
  }
  ScopedSetDevice(ScopedSetDevice const&) = delete;
  ScopedSetDevice& operator=(ScopedSetDevice const&) = delete;
  ~ScopedSetDevice() {
    if (restore_) static_cast<void>(cudaSetDevice(previous_));
  }

 private:
  int previous_ = -1;
  bool restore_ = false;
};

}  // namespace

void w4a16_mxfp4_latent_routed_moe_run(TensorView x, TensorView w13p, TensorView w13s,
                                       TensorView w2p, TensorView w2s, TensorView ids,
                                       TensorView tw, TensorView out) {
  int T = (int)x.shape()[0];
  const int device = x.device().device_id;
  ScopedSetDevice device_guard(device);
  cudaStream_t st = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, device));
  int MT = (T <= 64) ? 1 : MMM;
  size_t cap = (size_t)T * KTOP;

  size_t off_e = 0, off_m = 256, off_x = 512;
  size_t off_p = off_x + (cap + EL) * sizeof(int);
  size_t off_w = off_p + cap * sizeof(int);
  size_t off_h = off_w + cap * sizeof(float);
  size_t need = off_h + cap * IH * sizeof(__half);
  WorkspaceSlot* slot = nullptr;
  {
    std::lock_guard<std::mutex> registry(g_ws_registry_mu);
    for (auto& s : g_ws_slots) {
      if (s.used && s.device == device && s.stream == st) {
        slot = &s;
        break;
      }
    }
    if (!slot) {
      for (auto& s : g_ws_slots) {
        if (!s.used) {
          s.used = true;
          s.device = device;
          s.stream = st;
          slot = &s;
          break;
        }
      }
    }
  }
  void* fallback = nullptr;
  void* ws_ptr = nullptr;
  std::unique_lock<std::mutex> enqueue_guard;
  if (slot) {
    // Keep host threads from interleaving launches onto the same stream; GPU
    // stream order protects reuse after this complete enqueue sequence.
    enqueue_guard = std::unique_lock<std::mutex>(slot->enqueue_mu);
    if (need > slot->size) {
      // Reset the cached state BEFORE (re)allocating so a failed allocation
      // can never leave the slot claiming a workspace it does not hold.
      if (slot->ptr) {
        cudaError_t free_status = cudaFree(slot->ptr);
        slot->ptr = nullptr;
        slot->size = 0;
        check_cuda(free_status, "cudaFree(workspace)");
      } else {
        slot->size = 0;
      }
      void* grown = nullptr;
      check_cuda(cudaMalloc(&grown, need), "cudaMalloc(workspace)");
      slot->ptr = grown;
      slot->size = need;
    }
    ws_ptr = slot->ptr;
  } else {
    // An unusually large number of live streams falls back to a unique,
    // stream-ordered allocation rather than weakening correctness.
    check_cuda(cudaMallocAsync(&fallback, need, st), "cudaMallocAsync(workspace)");
    ws_ptr = fallback;
  }
  char* b = static_cast<char*>(ws_ptr);
  int* eoff = reinterpret_cast<int*>(b + off_e);
  int* moff = reinterpret_cast<int*>(b + off_m);
  int* mex = reinterpret_cast<int*>(b + off_x);
  int* ptok = reinterpret_cast<int*>(b + off_p);
  float* pw = reinterpret_cast<float*>(b + off_w);
  __half* hbuf = reinterpret_cast<__half*>(b + off_h);
  __nv_bfloat16* hbufb = reinterpret_cast<__nv_bfloat16*>(b + off_h);

  if (MT != 1) {
    cudaMemsetAsync(out.data_ptr(), 0, (size_t)T * LH * 2, st);
    route_k<false><<<1, 256, 0, st>>>(static_cast<const int*>(ids.data_ptr()),
                                      static_cast<const float*>(tw.data_ptr()), T, MT, eoff, moff,
                                      mex, ptok, pw, static_cast<__nv_bfloat16*>(out.data_ptr()));
  }

  if (MT == 1) {
    // The cooperative grid size is occupancy- and SM-count-derived, so it is
    // cached PER DEVICE ORDINAL: a second device with fewer SMs must not
    // inherit a grid the first device sized.
    static std::atomic<int> g_coop[64] = {};
    const bool cacheable = device >= 0 && device < 64;
    int coop = cacheable ? g_coop[device].load(std::memory_order_relaxed) : 0;
    if (coop == 0) {
      int nb = 0, nsm = 0;
      check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, moe_dec_k, 512,
                                                               (size_t)(LH * 2 + RCAP * 8)),
                 "cudaOccupancyMaxActiveBlocksPerMultiprocessor(moe_dec_k)");
      check_cuda(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, device),
                 "cudaDeviceGetAttribute(multiProcessorCount)");
      int g = nb * nsm;
      coop = (g < 1) ? 1 : (g > DGRID ? DGRID : g);
      if (cacheable) g_coop[device].store(coop, std::memory_order_relaxed);
    }
    const __nv_bfloat16* xp = static_cast<const __nv_bfloat16*>(x.data_ptr());
    const uint8_t* a1 = static_cast<const uint8_t*>(w13p.data_ptr());
    const uint8_t* a2 = static_cast<const uint8_t*>(w13s.data_ptr());
    const uint8_t* a3 = static_cast<const uint8_t*>(w2p.data_ptr());
    const uint8_t* a4 = static_cast<const uint8_t*>(w2s.data_ptr());
    const int* a5 = static_cast<const int*>(ids.data_ptr());
    const float* a6 = static_cast<const float*>(tw.data_ptr());
    __nv_bfloat16* a7 = static_cast<__nv_bfloat16*>(out.data_ptr());
    void* args[] = {(void*)&xp, (void*)&a1, (void*)&a2,   (void*)&a3, (void*)&a4,
                    (void*)&a5, (void*)&a6, (void*)&hbuf, (void*)&T,  (void*)&a7};
    check_cuda(cudaLaunchCooperativeKernel((const void*)moe_dec_k, dim3(coop), dim3(512), args,
                                           (size_t)(LH * 2 + RCAP * 8), st),
               "cudaLaunchCooperativeKernel(moe_dec_k)");
  } else {
    size_t sm1 = (size_t)(MMM + 2 * MNT) * MSTR * 2;
    size_t sm2 = (size_t)(MMM + 2 * MNT) * MSTR * 2;
    fc1_m_k<<<528, 256, sm1, st>>>(static_cast<const __nv_bfloat16*>(x.data_ptr()),
                                   static_cast<const uint8_t*>(w13p.data_ptr()),
                                   static_cast<const uint8_t*>(w13s.data_ptr()), eoff, moff, mex,
                                   ptok, hbufb);
    fc2_m_k<<<528, 256, sm2, st>>>(hbufb, static_cast<const uint8_t*>(w2p.data_ptr()),
                                   static_cast<const uint8_t*>(w2s.data_ptr()), eoff, moff, mex,
                                   ptok, pw, static_cast<__nv_bfloat16*>(out.data_ptr()));
  }
  if (fallback) cudaFreeAsync(fallback, st);
}

// Exported by csrc/flashinfer_fused_moe_w4a16_mxfp4_latent_sm90_binding.cu
