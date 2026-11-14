"""Race harness for the PDL ordering tests of #6799 (tests/utils/test_norm.py) and tests/trace/test_pdl_contract.py.

A producer releases its PDL dependents on entry, spins, and only then copies.
A dependent that reads before griddepcontrol.wait sees what the producer has
not written yet (RAW); one that writes before it changes the bytes the
producer copies out (WAR). The probe records whether a dependent launch
overlapped the producer at all, so the race tests skip instead of passing
vacuously where it never does.
"""

import os

import pytest
import torch

from flashinfer.utils import device_support_pdl

_PDL_RACE_CUDA = r"""
// Only the stream header, not ATen/cuda/CUDAContext.h: that one pulls in the
// cuBLAS / cuSPARSE headers, which a toolkit with just nvcc and cudart lacks.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

constexpr int kMaxJobs = 4;

struct CopyJobs {
  uint8_t* dst[kMaxJobs];
  const uint8_t* src[kMaxJobs];
  int64_t nbytes[kMaxJobs];
  int num_jobs;
};

__global__ void trigger_spin_copy_kernel(CopyJobs jobs, int64_t spin_cycles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  int64_t start = clock64();
  while (clock64() - start < spin_cycles) {
  }
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t tid = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  for (int j = 0; j < jobs.num_jobs; ++j) {
    for (int64_t i = tid; i < jobs.nbytes[j]; i += stride) {
      jobs.dst[j][i] = jobs.src[j][i];
    }
  }
}

__global__ void pdl_probe_kernel(const int32_t* flag, int32_t* observed) {
  // Read before griddepcontrol.wait on purpose: a stale value means this
  // dependent ran while the producer was still spinning.
  int32_t seen = *reinterpret_cast<const volatile int32_t*>(flag);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;");
#endif
  if (threadIdx.x == 0) {
    *observed = seen;
  }
}

void pdl_trigger_spin_copy(std::vector<at::Tensor> dsts, std::vector<at::Tensor> srcs,
                           int64_t spin_cycles) {
  TORCH_CHECK(dsts.size() == srcs.size(), "dsts and srcs differ in length");
  TORCH_CHECK(dsts.size() <= kMaxJobs, "too many copy jobs");
  CopyJobs jobs{};
  jobs.num_jobs = static_cast<int>(dsts.size());
  for (size_t j = 0; j < dsts.size(); ++j) {
    TORCH_CHECK(dsts[j].is_cuda() && srcs[j].is_cuda(), "copy jobs must be on CUDA");
    TORCH_CHECK(dsts[j].is_contiguous() && srcs[j].is_contiguous(),
                "copy jobs must be contiguous storage");
    int64_t n = dsts[j].numel() * dsts[j].element_size();
    TORCH_CHECK(n == srcs[j].numel() * srcs[j].element_size(), "copy job size mismatch");
    jobs.dst[j] = static_cast<uint8_t*>(dsts[j].data_ptr());
    jobs.src[j] = static_cast<const uint8_t*>(srcs[j].data_ptr());
    jobs.nbytes[j] = n;
  }
  trigger_spin_copy_kernel<<<64, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(jobs,
                                                                               spin_cycles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pdl_probe(at::Tensor flag, at::Tensor observed) {
  TORCH_CHECK(flag.is_cuda() && observed.is_cuda(), "probe tensors must be on CUDA");
  TORCH_CHECK(flag.scalar_type() == at::kInt && observed.scalar_type() == at::kInt,
              "probe tensors must be int32");
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(1);
  config.blockDim = dim3(32);
  config.dynamicSmemBytes = 0;
  config.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, pdl_probe_kernel,
                                    static_cast<const int32_t*>(flag.data_ptr()),
                                    static_cast<int32_t*>(observed.data_ptr())));
}
"""

PDL_SPIN_CYCLES = 4_000_000
# Raise for local stress runs; CI keeps the default.
PDL_REPEATS = int(os.environ.get("FLASHINFER_PDL_RACE_REPEATS", "5"))

_pdl_race_module_cache = {}


def pdl_race_module():
    major, minor = torch.cuda.get_device_capability()
    key = (major, minor)
    if key not in _pdl_race_module_cache:
        from torch.utils.cpp_extension import load_inline

        _pdl_race_module_cache[key] = load_inline(
            name=f"test_pdl_race_sm{major}{minor}",
            cpp_sources=[
                "void pdl_trigger_spin_copy(std::vector<at::Tensor> dsts, "
                "std::vector<at::Tensor> srcs, int64_t spin_cycles);\n"
                "void pdl_probe(at::Tensor flag, at::Tensor observed);"
            ],
            cuda_sources=[_PDL_RACE_CUDA],
            extra_cuda_cflags=[
                f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
            ],
            functions=["pdl_trigger_spin_copy", "pdl_probe"],
            verbose=False,
        )
        module = _pdl_race_module_cache[key]
        # Load both kernels eagerly: a lazily loaded module cannot be loaded
        # while a CUDA graph is being captured.
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        observed = torch.zeros(1, dtype=torch.int32, device="cuda")
        module.pdl_trigger_spin_copy([], [], 0)
        module.pdl_probe(flag, observed)
        torch.cuda.synchronize()
    return _pdl_race_module_cache[key]


def skip_unless_pdl():
    if not device_support_pdl(torch.device("cuda")):
        # enable_pdl=True is turned into a plain launch below SM90, which would
        # make every race test pass without testing anything.
        pytest.skip("PDL requires SM90+")


_pdl_overlap_cache = {}


def _pdl_overlap_runs(mode):
    """How many of ``PDL_REPEATS`` probe runs started a PDL dependent while
    the producer was still spinning."""
    key = (torch.cuda.current_device(), mode)
    if key not in _pdl_overlap_cache:
        module = pdl_race_module()
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        one = torch.ones(1, dtype=torch.int32, device="cuda")
        observed = torch.full((1,), -1, dtype=torch.int32, device="cuda")

        def fn():
            module.pdl_trigger_spin_copy([flag], [one], PDL_SPIN_CYCLES)
            module.pdl_probe(flag, observed)

        run = pdl_launcher(mode, fn)
        stale = 0
        for _ in range(PDL_REPEATS):
            flag.zero_()
            observed.fill_(-1)
            torch.cuda.synchronize()
            run()
            torch.cuda.synchronize()
            assert int(flag.item()) == 1
            assert int(observed.item()) in (0, 1)
            stale += int(observed.item()) == 0
        _pdl_overlap_cache[key] = stale
    return _pdl_overlap_cache[key]


def skip_unless_pdl_overlaps(mode):
    """Skip where a PDL dependent cannot overlap the producer: the race tests
    would pass there without testing anything."""
    skip_unless_pdl()
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("", "0"):
        pytest.skip("CUDA_LAUNCH_BLOCKING serializes launches; PDL cannot overlap")
    if _pdl_overlap_runs(mode) == 0:
        # Something serialized the launches, e.g. a sanitizer or profiler.
        pytest.skip(
            f"{mode}: no PDL dependent ran ahead of the producer in "
            f"{PDL_REPEATS} probe runs, so the race tests cannot fail here"
        )


_BITS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _bits(t):
    return t.view(_BITS[t.element_size()])


def assert_bitwise(actual, expected, what):
    a, e = _bits(actual), _bits(expected)
    bad = a != e
    if bad.any():
        rows = bad.reshape(-1, bad.shape[-1]).any(dim=-1).nonzero().flatten()
        raise AssertionError(
            f"{what}: {int(bad.sum())} of {bad.numel()} elements differ bitwise; "
            f"first differing flattened rows {rows[:8].tolist()}"
        )


def pdl_launcher(mode, fn):
    """Return a callable that runs ``fn`` eagerly or replays it from a CUDA graph."""
    if mode == "eager":
        return fn
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay
