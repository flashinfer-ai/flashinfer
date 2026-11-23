"""Time a kernel inside one CUDA graph, with a device-side clock wait before every sample.

The graph holds ``n`` blocks ``[wait_clock, L2 flush, stamp, kernel, stamp]``. The wait kernel
spins on one thread until the SM clock (clock64 cycles per globaltimer ns) is back at its
maximum, so no sample starts on a throttled GPU and the host is out of the loop. The two stamp
kernels write the 32 ns resolution globaltimer; a sample is the stamp-to-stamp time minus the
stamp-to-stamp time of an empty kernel (the fixed launch gaps of the stamps).
"""

import os
import statistics

import torch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

// out[slot*4 + {0..3}] = waited_ns, last MHz, reached (1/0), windows
__global__ void wait_clock_kernel(unsigned long long window_ns, unsigned long long timeout_ns,
                                  long long min_mhz, int need, long long* out, int slot) {
  if (threadIdx.x != 0) return;
  unsigned long long start = gtimer();
  int good = 0, windows = 0;
  long long mhz = 0;
  while (true) {
    unsigned long long t0 = gtimer();
    long long c0 = clock64();
    unsigned long long t1;
    do { t1 = gtimer(); } while (t1 - t0 < window_ns);
    long long c1 = clock64();
    mhz = (long long)(((unsigned long long)(c1 - c0)) * 1000ull / (t1 - t0));
    windows++;
    if (mhz >= min_mhz) { if (++good >= need) break; } else { good = 0; }
    if (t1 - start > timeout_ns) break;
  }
  long long* o = out + (long long)slot * 4;
  o[0] = (long long)(gtimer() - start);
  o[1] = mhz;
  o[2] = (good >= need) ? 1 : 0;
  o[3] = windows;
}
__global__ void stamp_kernel(unsigned long long* out) {
  if (threadIdx.x == 0) *out = gtimer();
}
__global__ void noop_kernel() {}

void wait_clock(at::Tensor out, int64_t slot, int64_t window_ns, int64_t timeout_ns, int64_t min_mhz, int64_t need) {
  auto stream = at::cuda::getCurrentCUDAStream();
  wait_clock_kernel<<<1, 32, 0, stream>>>((unsigned long long)window_ns, (unsigned long long)timeout_ns,
      (long long)min_mhz, (int)need, reinterpret_cast<long long*>(out.data_ptr<int64_t>()), (int)slot);
}
void stamp(at::Tensor out, int64_t idx) {
  auto stream = at::cuda::getCurrentCUDAStream();
  stamp_kernel<<<1, 32, 0, stream>>>(reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()) + idx);
}
void noop() {
  auto stream = at::cuda::getCurrentCUDAStream();
  noop_kernel<<<1, 32, 0, stream>>>();
}
"""
_CPP_SRC = """
void wait_clock(at::Tensor out, int64_t slot, int64_t window_ns, int64_t timeout_ns, int64_t min_mhz, int64_t need);
void stamp(at::Tensor out, int64_t idx);
void noop();
"""
_mod = None
_overhead_us: dict[int, float] = {}


def load():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        cap = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}")
        build = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.expanduser(
            "~/.cache/flashinfer_graph_wait"
        )
        os.makedirs(build, exist_ok=True)
        _mod = load_inline(
            "flashinfer_graph_wait",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["wait_clock", "stamp", "noop"],
            build_directory=build,
            verbose=False,
        )
    return _mod


def stamp_overhead_us(n: int = 10) -> float:
    """Stamp-to-stamp time of an empty kernel: the fixed cost of the two stamps."""
    dev = torch.cuda.current_device()
    if dev not in _overhead_us:
        ext = load()
        stamps = torch.zeros(2 * n, dtype=torch.int64, device="cuda")
        s = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for i in range(n):
                ext.stamp(stamps, 2 * i)
                ext.noop()
                ext.stamp(stamps, 2 * i + 1)
        vals = []
        for _ in range(10):
            g.replay()
            torch.cuda.synchronize()
            st = stamps.view(n, 2).cpu()
            vals += ((st[:, 1] - st[:, 0]).float() / 1e3).tolist()
        _overhead_us[dev] = statistics.median(vals)
    return _overhead_us[dev]


def measure_series(
    run_once,
    batches,
    l2_flush,
    n: int,
    min_mhz: int,
    timeout_ns: int,
    window_ns: int = 100_000,
    need: int = 2,
    max_attempts: int = 3,
):
    """Return (samples_us, waits, retries, ok, overhead_us); one sample per block."""
    ext = load()
    overhead = stamp_overhead_us()
    stamps = torch.zeros(2 * n, dtype=torch.int64, device="cuda")
    waits = torch.zeros(4 * n, dtype=torch.int64, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n):
            ext.wait_clock(waits, i, window_ns, timeout_ns, min_mhz, need)
            l2_flush()
            ext.stamp(stamps, 2 * i)
            run_once(batches[i % len(batches)])
            ext.stamp(stamps, 2 * i + 1)
    g.replay()  # untimed: first replay of a new graph is slower
    torch.cuda.synchronize()
    retries = 0
    for attempt in range(max_attempts):
        g.replay()
        torch.cuda.synchronize()
        w = waits.view(n, 4).cpu()
        ok = bool(w[:, 2].all()) if min_mhz > 0 else True
        retries = attempt
        if ok:
            break
    st = stamps.view(n, 2).cpu()
    samples = ((st[:, 1] - st[:, 0]).float() / 1e3 - overhead).tolist()
    return samples, w.tolist(), retries, ok, overhead
