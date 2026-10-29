"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F
from tests.test_helpers.parametrize import pairwise_product_cases, parametrize_product
from tests.test_helpers.pdl import late_store

import flashinfer
from flashinfer.jit import env as jit_env
from flashinfer.jit.core import gen_jit_spec
from flashinfer.utils import device_support_pdl


def _skip_if_fused_norm_smem_exceeds_limit(hidden_size, dtype, device):
    # Match FusedAddRMSNorm{,Quant} and GemmaFusedAddRMSNorm in norm.cuh:
    # one FP32 row plus the warp-reduction scratch space. A 16384-wide row
    # needs 65664 bytes, exceeding T4's 65536-byte opt-in per-block limit.
    vec_size = math.gcd(16 // dtype.itemsize, hidden_size)
    num_warps = (min(1024, hidden_size // vec_size) + 31) // 32
    required = (hidden_size + ((num_warps + 3) // 4) * 4) * 4
    limit = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    if required > limit:
        pytest.skip(
            f"Fused norm requires {required} bytes of shared memory per block, "
            f"but this device supports {limit}"
        )


def llama_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def llama_rms_norm_quant(x, w, scale, quant_dtype, eps=1e-6):
    inv_scale = torch.reciprocal(torch.tensor(scale)).float()
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x * inv_scale
    finfo = torch.finfo(quant_dtype)
    x = torch.clamp(x, finfo.min, finfo.max)
    x = x.to(quant_dtype)
    return x


def layer_norm_quant(x, gamma, beta, scale, quant_dtype, eps=1e-6, round_to_input=True):
    y = torch.nn.functional.layer_norm(
        x.float(), (x.shape[-1],), weight=gamma, bias=beta, eps=eps
    )
    # The CUDA kernel rounds to input dtype before scaling; the CuTe-DSL
    # kernel stays in fp32.
    if round_to_input:
        y = y.to(x.dtype)
    y = y.float() / scale
    finfo = torch.finfo(quant_dtype)
    y = torch.clamp(y, finfo.min, finfo.max)
    return y.to(quant_dtype)


def gemma_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x


def gemma_fused_add_rms_norm(x, residual, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x + residual
    residual = x
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x, residual


def fused_add_rms_norm(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * weight.float()).to(orig_dtype)
    return x, residual


def fused_add_rms_norm_quant(x, residual, weight, scale, quant_dtype, eps):
    inv_scale = torch.reciprocal(torch.tensor(scale)).float()
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * weight.float()
    x = x * inv_scale
    finfo = torch.finfo(quant_dtype)
    x = torch.clamp(x, finfo.min, finfo.max)
    x = x.to(quant_dtype)
    return x, residual


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_norm(batch_size, hidden_size, dtype, specify_out, enable_pdl, contiguous):
    if contiguous:
        x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = llama_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        flashinfer.norm.rmsnorm(x, w, out=y, enable_pdl=enable_pdl)
    else:
        y = flashinfer.norm.rmsnorm(x, w, enable_pdl=enable_pdl)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(5, 111), (5, 1024), (3, 5, 64), (3, 5, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("pattern", ["zero", "tiny", "sparse", "alternating", "large"])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_rmsnorm_output_contract(shape, dtype, padded, pattern, enable_pdl):
    """Exercise finite edge inputs and caller-owned output storage on the public API."""
    if enable_pdl and not device_support_pdl(torch.device("cuda")):
        # rmsnorm() silently turns enable_pdl=True into a plain launch below SM90;
        # skip instead of counting that launch as PDL coverage.
        pytest.skip("PDL requires SM90+")
    hidden = shape[-1]
    storage_shape = (*shape[:-1], hidden + 16) if padded else shape
    input_storage = torch.full(storage_shape, -23, dtype=dtype, device="cuda")
    output_storage = torch.full_like(input_storage, -23)
    x = input_storage[..., 8:-8] if padded else input_storage
    out = output_storage[..., 8:-8] if padded else output_storage
    columns = torch.arange(hidden, device="cuda")
    signs = (columns % 2 * 2 - 1).to(dtype)
    if pattern == "zero":
        x.zero_()
    elif pattern == "tiny":
        x.copy_(signs * 2**-14)
    elif pattern == "sparse":
        x.zero_()
        x[..., hidden // 2] = 2
    elif pattern == "alternating":
        x.copy_(signs)
    else:
        # Finite in the input dtype and FP32 reduction, but not an FP16 square.
        magnitude = 2**14 if dtype == torch.float16 else 2**30
        x.copy_(signs * magnitude)
    weight = ((columns % 9 - 4).float() / 4).to(dtype)
    input_before = input_storage.clone()
    weight_before = weight.clone()
    expected = llama_rms_norm(x, weight)
    assert torch.isfinite(expected).all()
    tolerance = 1e-3 if dtype == torch.float16 else 1e-2

    previous = None
    for poison in (float("nan"), 17.0):
        out.fill_(poison)
        returned = flashinfer.norm.rmsnorm(x, weight, out=out, enable_pdl=enable_pdl)
        torch.cuda.synchronize()
        assert returned is out
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(input_storage, input_before, rtol=0, atol=0)
        torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)
        if padded:
            assert (output_storage[..., :8] == -23).all()
            assert (output_storage[..., -8:] == -23).all()
        if previous is not None:
            torch.testing.assert_close(out, previous, rtol=0, atol=0)
        previous = out.clone()


# ---------------------------------------------------------------------------
# PDL ordering. With enable_pdl the norm kernels are launched with programmatic
# stream serialization, so they may start while the previous kernel is still
# running. They must not read what that kernel is still writing (RAW) or write
# what it is still reading (WAR) before griddepcontrol.wait.
# ---------------------------------------------------------------------------

# The producer releases its dependents on entry, spins, and only then copies.
# A dependent that reads before griddepcontrol.wait sees the NaN left in a
# destination (RAW); one that writes before it changes the bytes the producer
# copies out (WAR). The probe reads a flag before its own wait, so it records
# whether a dependent launch overlapped the producer at all on this setup.
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

_PDL_SPIN_CYCLES = 4_000_000
# Raise for local stress runs; CI keeps the default.
_PDL_REPEATS = int(os.environ.get("FLASHINFER_NORM_PDL_REPEATS", "5"))
_PDL_SEED = 3605

# (api, shape, padded): one entry per CuTe variant the SM90 dispatch can pick,
# see the cluster / cp.async / compact-vs-strided selection in
# flashinfer/norm/kernels/{rmsnorm,fused_add_rmsnorm}.py. M = 264 keeps every
# SM of a 132-SM part busy twice over.
#
# Weights are deliberately not part of the contract: they are static
# parameters, and reading them before griddepcontrol.wait (for example to
# prefetch them) is a legitimate optimisation. The producer therefore only
# writes or reads activations, the residual and the output.
_PDL_RACE_CASES = [
    ("rmsnorm", (264, 128), False),
    ("rmsnorm", (264, 4096), False),
    ("rmsnorm", (264, 4096), True),
    ("rmsnorm", (264, 131072), False),
    ("rmsnorm", (66, 4, 128), False),
    ("fused_add_rmsnorm", (264, 128), False),
    ("fused_add_rmsnorm", (264, 4096), False),
    ("fused_add_rmsnorm", (264, 32768), False),
]


def _pdl_case_id(case):
    api, shape, padded = case
    return f"{api}-{'x'.join(map(str, shape))}{'-padded' if padded else ''}"


_pdl_race_module_cache = {}


def _pdl_race_module():
    major, minor = torch.cuda.get_device_capability()
    key = (major, minor)
    if key not in _pdl_race_module_cache:
        from torch.utils.cpp_extension import load_inline

        _pdl_race_module_cache[key] = load_inline(
            name=f"test_norm_pdl_race_sm{major}{minor}",
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


def _skip_unless_pdl():
    if not device_support_pdl(torch.device("cuda")):
        # enable_pdl=True is turned into a plain launch below SM90, which would
        # make every race test pass without testing anything.
        pytest.skip("PDL requires SM90+")


_pdl_overlap_cache = {}


def _pdl_overlap_runs(mode):
    """How many of ``_PDL_REPEATS`` probe runs started a PDL dependent while
    the producer was still spinning."""
    key = (torch.cuda.current_device(), mode)
    if key not in _pdl_overlap_cache:
        module = _pdl_race_module()
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        one = torch.ones(1, dtype=torch.int32, device="cuda")
        observed = torch.full((1,), -1, dtype=torch.int32, device="cuda")

        def fn():
            module.pdl_trigger_spin_copy([flag], [one], _PDL_SPIN_CYCLES)
            module.pdl_probe(flag, observed)

        run = _pdl_launcher(mode, fn)
        stale = 0
        for _ in range(_PDL_REPEATS):
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


def _skip_unless_pdl_overlaps(mode):
    """Skip where a PDL dependent cannot overlap the producer: the race tests
    would pass there without testing anything."""
    _skip_unless_pdl()
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("", "0"):
        pytest.skip("CUDA_LAUNCH_BLOCKING serializes launches; PDL cannot overlap")
    if _pdl_overlap_runs(mode) == 0:
        # Something serialized the launches, e.g. a sanitizer or profiler.
        pytest.skip(
            f"{mode}: no PDL dependent ran ahead of the producer in "
            f"{_PDL_REPEATS} probe runs, so the race tests cannot fail here"
        )


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _assert_bitwise(actual, expected, what):
    a, e = _bits(actual), _bits(expected)
    bad = a != e
    if bad.any():
        rows = bad.reshape(-1, bad.shape[-1]).any(dim=-1).nonzero().flatten()
        raise AssertionError(
            f"{what}: {int(bad.sum())} of {bad.numel()} elements differ bitwise; "
            f"first differing flattened rows {rows[:8].tolist()}"
        )


def _pdl_launcher(mode, fn):
    """Return a callable that runs ``fn`` eagerly or replays it from a CUDA graph."""
    if mode == "eager":
        return fn
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


class _PdlNormCase:
    """Storage for one race case. Every tensor the producer touches is the
    contiguous storage; the norm sees the (possibly padded) view."""

    def __init__(self, api, shape, padded, dtype):
        self.api, self.shape, self.padded, self.dtype = api, shape, padded, dtype
        hidden = shape[-1]
        self.storage_shape = (*shape[:-1], hidden + 16) if padded else shape
        gen = torch.Generator(device="cuda").manual_seed(_PDL_SEED)
        self.src_x = torch.randn(
            self.storage_shape, generator=gen, device="cuda", dtype=torch.float32
        ).to(dtype)
        self.src_r = torch.randn(
            self.storage_shape, generator=gen, device="cuda", dtype=torch.float32
        ).to(dtype)
        self.src_w = (
            torch.rand(hidden, generator=gen, device="cuda", dtype=torch.float32) + 0.5
        ).to(dtype)
        self.x = torch.empty_like(self.src_x)
        self.r = torch.empty_like(self.src_r)
        self.w = torch.empty_like(self.src_w)
        self.out = torch.empty_like(self.src_x)

    def view(self, t):
        return t[..., 8:-8] if self.padded else t

    def call(self, enable_pdl):
        if self.api == "rmsnorm":
            flashinfer.norm.rmsnorm(
                self.view(self.x),
                self.w,
                out=self.view(self.out),
                enable_pdl=enable_pdl,
            )
        else:
            flashinfer.norm.fused_add_rmsnorm(
                self.view(self.x), self.view(self.r), self.w, enable_pdl=enable_pdl
            )

    def reset(self):
        """Clean inputs, poisoned output; padding keeps a known value."""
        self.x.copy_(self.src_x)
        self.r.copy_(self.src_r)
        self.w.copy_(self.src_w)
        self.out.fill_(17.0)

    def results(self):
        if self.api == "rmsnorm":
            return {"out": self.out.clone()}
        return {"input": self.x.clone(), "residual": self.r.clone()}

    def reference(self, enable_pdl):
        """The same call on synchronized inputs: the bitwise target."""
        self.reset()
        torch.cuda.synchronize()
        self.call(enable_pdl)
        torch.cuda.synchronize()
        return self.results()


def _check_shape_fits(case):
    api, shape, _ = case
    if api == "fused_add_rmsnorm":
        _skip_if_fused_norm_smem_exceeds_limit(
            shape[-1], torch.bfloat16, torch.device("cuda")
        )


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_pdl_race_probe_sees_overlap(mode):
    """Calibration for the race tests below: they can only fail where a PDL
    dependent actually starts while the producer spins. Like them, this skips
    with the reason on setups where it never does."""
    _skip_unless_pdl_overlaps(mode)


@pytest.mark.parametrize("case", _PDL_RACE_CASES, ids=_pdl_case_id)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_norm_pdl_waits_before_reading(case, dtype, mode):
    """RAW: activations written by a still-running producer must not be read
    early. Weights are excluded, see ``_PDL_RACE_CASES``."""
    _skip_unless_pdl_overlaps(mode)
    _check_shape_fits(case)
    module = _pdl_race_module()
    c = _PdlNormCase(*case, dtype)
    expected = c.reference(enable_pdl=True)

    dsts, srcs = [c.x], [c.src_x]
    if c.api == "fused_add_rmsnorm":
        dsts.append(c.r)
        srcs.append(c.src_r)

    def fn():
        module.pdl_trigger_spin_copy(dsts, srcs, _PDL_SPIN_CYCLES)
        c.call(enable_pdl=True)

    run = _pdl_launcher(mode, fn)
    for rep in range(_PDL_REPEATS):
        c.reset()
        for t in dsts:
            t.fill_(float("nan"))
        torch.cuda.synchronize()
        run()
        torch.cuda.synchronize()
        for name, value in c.results().items():
            _assert_bitwise(
                value,
                expected[name],
                f"{_pdl_case_id(case)} {dtype} {mode} "
                f"seed={_PDL_SEED} repeat={rep} {name}",
            )


@pytest.mark.parametrize("case", _PDL_RACE_CASES, ids=_pdl_case_id)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_norm_pdl_waits_before_writing(case, dtype, mode):
    """WAR: buffers a still-running producer reads must not be overwritten early."""
    _skip_unless_pdl_overlaps(mode)
    _check_shape_fits(case)
    module = _pdl_race_module()
    c = _PdlNormCase(*case, dtype)
    expected = c.reference(enable_pdl=True)

    if c.api == "rmsnorm":
        watched = {"out": c.out}
    else:
        watched = {"input": c.x, "residual": c.r}
    snapshots = {name: torch.empty_like(t) for name, t in watched.items()}

    def fn():
        module.pdl_trigger_spin_copy(
            list(snapshots.values()), list(watched.values()), _PDL_SPIN_CYCLES
        )
        c.call(enable_pdl=True)

    run = _pdl_launcher(mode, fn)
    for rep in range(_PDL_REPEATS):
        c.reset()
        before = {name: t.clone() for name, t in watched.items()}
        for s in snapshots.values():
            s.fill_(-1.0)
        torch.cuda.synchronize()
        run()
        torch.cuda.synchronize()
        tag = f"{_pdl_case_id(case)} {dtype} {mode} seed={_PDL_SEED} repeat={rep}"
        for name in watched:
            _assert_bitwise(
                snapshots[name], before[name], f"{tag} {name} overwritten early"
            )
        for name, value in c.results().items():
            _assert_bitwise(value, expected[name], f"{tag} {name}")


@pytest.mark.parametrize("case", _PDL_RACE_CASES, ids=_pdl_case_id)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_norm_pdl_matches_plain_launch(case, dtype):
    """The PDL and non-PDL builds of a kernel give bitwise identical results."""
    _skip_unless_pdl()
    _check_shape_fits(case)
    c = _PdlNormCase(*case, dtype)
    with_pdl = c.reference(enable_pdl=True)
    without_pdl = c.reference(enable_pdl=False)
    for name, value in with_pdl.items():
        _assert_bitwise(
            value, without_pdl[name], f"{_pdl_case_id(case)} {dtype} {name}"
        )


@parametrize_product(
    {
        "batch_size": [1, 19, 99, 989],
        "hidden_size": [111, 500, 1024, 3072, 3584, 4096, 8192, 16384],
        "dtype": [torch.float16, torch.bfloat16],
        "quant_dtype": [torch.float8_e4m3fn, torch.float8_e5m2],
        "quant_scale": [0.01, 1.0, 10.0],
        "enable_pdl": [True, False],
        "contiguous": [True, False],
    },
    regular=pairwise_product_cases,
)
def test_norm_quant(
    batch_size, hidden_size, dtype, quant_dtype, quant_scale, enable_pdl, contiguous
):
    if contiguous:
        x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = llama_rms_norm_quant(x, w, quant_scale, quant_dtype)
    y = torch.empty_like(x, dtype=quant_dtype, device="cuda")
    flashinfer.norm.rmsnorm_quant(
        y, x, w, torch.tensor(quant_scale, device="cuda"), enable_pdl=enable_pdl
    )

    torch.testing.assert_close(y_ref.float(), y.float(), rtol=1, atol=1)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("num_heads", [4, 7, 16])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_qknorm(
    batch_size, num_heads, head_dim, dtype, specify_out, enable_pdl, contiguous
):
    if contiguous:
        x = torch.randn(batch_size, num_heads, head_dim).to(0).to(dtype)
    else:
        x = torch.randn(batch_size, num_heads * 2, head_dim, device="cuda").to(dtype)
        x = x[:, :num_heads, :head_dim]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    w = torch.randn(head_dim).to(0).to(dtype)

    y_ref = llama_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        flashinfer.norm.rmsnorm(x, w, out=y, enable_pdl=enable_pdl)
    else:
        y = flashinfer.norm.rmsnorm(x, w, enable_pdl=enable_pdl)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_fused_add_rmsnorm(batch_size, hidden_size, dtype, enable_pdl, contiguous):
    _skip_if_fused_norm_smem_exceeds_limit(hidden_size, dtype, "cuda")

    eps = 1e-6

    if contiguous:
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    flashinfer.fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


@parametrize_product(
    {
        "batch_size": [1, 19, 99, 989],
        "hidden_size": [111, 500, 1024, 3072, 3584, 4096, 8192, 16384],
        "dtype": [torch.float16, torch.bfloat16],
        "quant_dtype": [torch.float8_e4m3fn, torch.float8_e5m2],
        "quant_scale": [0.01, 1.0, 10.0],
        "enable_pdl": [True, False],
        "contiguous": [True, False],
    },
    regular=pairwise_product_cases,
)
def test_fused_add_rmsnorm_quant(
    batch_size, hidden_size, dtype, quant_dtype, quant_scale, enable_pdl, contiguous
):
    _skip_if_fused_norm_smem_exceeds_limit(hidden_size, dtype, "cuda")

    eps = 1e-6

    if contiguous:
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = fused_add_rms_norm_quant(
        x.clone(), residual.clone(), weight, quant_scale, quant_dtype, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    y = torch.empty_like(x, dtype=quant_dtype, device="cuda")
    flashinfer.norm.fused_add_rmsnorm_quant(
        y,
        x_fused,
        residual_fused,
        weight,
        torch.tensor(quant_scale, device="cuda"),
        eps,
        enable_pdl=enable_pdl,
    )

    torch.testing.assert_close(y.float(), x_native.float(), rtol=1, atol=1)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


def fused_add_rms_norm_fp8_block_quant(x, residual, weight, eps, block=128):
    """Reference: add residual, RMSNorm, then per-1x128-block fp8-e4m3 quant.

    Returns (fp8 out, scale[M, H/block], pre-norm residual, bf16/fp16 normed).
    """
    orig_dtype = x.dtype
    xf = x.to(torch.float32) + residual.to(torch.float32)
    residual = xf.to(orig_dtype)  # pre-norm residual
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = (xf * torch.rsqrt(variance + eps) * weight.float()).to(
        orig_dtype
    )  # round to T
    M, H = normed.shape
    nb = normed.float().view(M, H // block, block)
    amax = nb.abs().amax(dim=-1).clamp_min(1e-4)  # (M, H/block)
    scale = amax / 448.0  # e4m3 max
    q = (nb / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(M, H)
    return q, scale, residual, normed


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [256, 512, 1024, 3072, 4096, 6144, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_fused_add_rmsnorm_fp8_block_quant(batch_size, hidden_size, dtype, enable_pdl):
    eps = 1e-6
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda") * 0.1
    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")
    residual = torch.randn_like(x) * 0.1
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    q_ref, scale_ref, residual_ref, normed_ref = fused_add_rms_norm_fp8_block_quant(
        x.clone(), residual.clone(), weight, eps
    )

    m_pad = (batch_size + 3) & ~3
    out = torch.empty(batch_size, hidden_size, dtype=torch.float8_e4m3fn, device="cuda")
    block_scale = torch.empty(
        hidden_size // 128, m_pad, dtype=torch.float32, device="cuda"
    )
    normed_out = torch.empty_like(x)
    residual_fused = residual.clone()
    flashinfer.norm.fused_add_rmsnorm_fp8_block_quant(
        out,
        block_scale,
        normed_out,
        x,
        residual_fused,
        weight,
        eps,
        enable_pdl=enable_pdl,
    )

    # pre-norm residual and bf16/fp16 normed: tight
    torch.testing.assert_close(residual_fused, residual_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(normed_out, normed_ref, rtol=1e-2, atol=1e-2)
    # per-1x128-block scale, read back from the (H/128, round_up(M,4)) column-major buffer.
    # scale = block amax / 448; amax can differ by up to ~2 low-dtype ULP between kernel and
    # reference (fast rsqrt + a different reduction order flip the max element's rounding), so
    # use a dtype-appropriate rtol rather than an exact match.
    scale_got = block_scale.transpose(0, 1)[:batch_size]
    torch.testing.assert_close(scale_got, scale_ref, rtol=2e-2, atol=1e-5)
    # dequantized fp8 vs the reference normed (loose fp8 tolerance)
    deq = out.float().view(batch_size, hidden_size // 128, 128) * scale_got.unsqueeze(
        -1
    )
    torch.testing.assert_close(
        deq.view(batch_size, hidden_size), normed_ref.float(), rtol=0.1, atol=0.3
    )


@pytest.mark.parametrize("hidden_size", [1152, 32768])
def test_fused_add_rmsnorm_fp8_block_quant_rejects_unsupported_hidden(hidden_size):
    # single-wave contract: hidden_size must be <= 16384 and a multiple of 32*vec_size
    # (256 for <=8192). 1152 is %128==0 but %256!=0; 32768 exceeds 16384. Both must raise.
    batch_size = 64
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
    m_pad = (batch_size + 3) & ~3
    out = torch.empty(batch_size, hidden_size, dtype=torch.float8_e4m3fn, device="cuda")
    block_scale = torch.empty(
        hidden_size // 128, m_pad, dtype=torch.float32, device="cuda"
    )
    normed_out = torch.empty_like(x)
    with pytest.raises(RuntimeError, match="hidden_size"):
        flashinfer.norm.fused_add_rmsnorm_fp8_block_quant(
            out, block_scale, normed_out, x, residual, weight, 1e-6, enable_pdl=False
        )


@pytest.mark.parametrize(
    "bad",
    [
        "out_dtype",
        "out_shape",
        "block_scale_dtype",
        "block_scale_stride",
        "normed_out_dtype",
    ],
)
def test_fused_add_rmsnorm_fp8_block_quant_rejects_bad_outputs(bad):
    # The kernel reinterprets the output buffers with fixed types (e4m3 out, fp32 block_scale,
    # input-dtype normed_out) and indexes block_scale unit-stride along m, so each of these would
    # silently corrupt memory without a host-side check.
    batch_size, hidden_size = 64, 1024
    dtype = torch.bfloat16
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
    m_pad = (batch_size + 3) & ~3
    out = torch.empty(batch_size, hidden_size, dtype=torch.float8_e4m3fn, device="cuda")
    block_scale = torch.empty(
        hidden_size // 128, m_pad, dtype=torch.float32, device="cuda"
    )
    normed_out = torch.empty_like(x)

    if bad == "out_dtype":
        out = out.to(torch.float8_e5m2)
    elif bad == "out_shape":
        out = out[:, : hidden_size // 2]
    elif bad == "block_scale_dtype":
        block_scale = block_scale.to(torch.float16)
    elif bad == "block_scale_stride":
        # transposed (m-major) view: last dim is no longer unit-stride
        block_scale = torch.empty(
            m_pad, hidden_size // 128, dtype=torch.float32, device="cuda"
        ).t()
    elif bad == "normed_out_dtype":
        normed_out = normed_out.to(torch.float32)

    with pytest.raises(RuntimeError):
        flashinfer.norm.fused_add_rmsnorm_fp8_block_quant(
            out, block_scale, normed_out, x, residual, weight, 1e-6, enable_pdl=False
        )


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_gemma_norm(
    batch_size, hidden_size, dtype, specify_out, enable_pdl, contiguous
):
    if contiguous:
        x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = gemma_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        flashinfer.norm.gemma_rmsnorm(x, w, out=y, enable_pdl=enable_pdl)
    else:
        y = flashinfer.norm.gemma_rmsnorm(x, w, enable_pdl=enable_pdl)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_gemma_fused_add_rmsnorm(
    batch_size, hidden_size, dtype, enable_pdl, contiguous
):
    _skip_if_fused_norm_smem_exceeds_limit(hidden_size, dtype, "cuda")

    eps = 1e-6

    if contiguous:
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = gemma_fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    flashinfer.gemma_fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 128])
@pytest.mark.parametrize("hidden_size", [128, 129, 1024, 16384])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_layernorm(batch_size, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    gamma = torch.randn(hidden_size, dtype=torch.float32, device="cuda")
    beta = torch.randn(hidden_size, dtype=torch.float32, device="cuda")

    out = flashinfer.layernorm(x, gamma, beta, eps)
    out_ref = F.layer_norm(x.float(), (hidden_size,), gamma, beta, eps).to(dtype)

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 4096, 8192, 16384])
@pytest.mark.parametrize("quant_scale", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_layernorm_quant(
    batch_size, hidden_size, quant_scale, quant_dtype, backend, monkeypatch
):
    if backend == "cuda":
        monkeypatch.setattr(flashinfer.norm, "_USE_CUDA_NORM", True)
    elif not hasattr(flashinfer.norm, "layernorm_quant_cute"):
        pytest.skip("nvidia-cutlass-dsl not available")
    elif not flashinfer.norm._cute_dsl_supports_arch(
        *torch.cuda.get_device_capability()
    ):
        pytest.skip("CuTe DSL cannot target this compute capability")
    else:
        # Dispatch also honors FLASHINFER_USE_CUDA_NORM from the environment;
        # pin it so this branch really exercises the DSL kernel.
        monkeypatch.setattr(flashinfer.norm, "_use_cuda_norm", lambda _device: False)

    # The check bounds the fraction of drifting elements, so fix the seed.
    torch.manual_seed(0)
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="cuda")
    gamma = torch.randn(hidden_size, dtype=torch.float32, device="cuda")
    beta = torch.randn(hidden_size, dtype=torch.float32, device="cuda")
    scale = torch.tensor([quant_scale], dtype=torch.float32, device="cuda")

    y_ref = layer_norm_quant(
        x, gamma, beta, scale, quant_dtype, round_to_input=(backend == "cuda")
    )
    y = torch.empty_like(x, dtype=quant_dtype)
    flashinfer.norm.layernorm_quant(y, x, gamma, beta, scale)

    # fp8 defaults (rtol/atol=0.1) allow a one-bucket step; max_mismatch_pct
    # bounds how many elements may drift, so a systematic error still fails.
    from flashinfer.trace import default_check

    assert default_check([y_ref], [y], max_mismatch_pct=1.0, min_cos_sim=None)


@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_layernorm_quant_invalid_inputs(backend, monkeypatch):
    if backend == "cuda":
        monkeypatch.setattr(flashinfer.norm, "_USE_CUDA_NORM", True)
    elif not hasattr(flashinfer.norm, "layernorm_quant_cute"):
        pytest.skip("nvidia-cutlass-dsl not available")
    elif not flashinfer.norm._cute_dsl_supports_arch(
        *torch.cuda.get_device_capability()
    ):
        pytest.skip("CuTe DSL cannot target this compute capability")
    else:
        # Dispatch also honors FLASHINFER_USE_CUDA_NORM from the environment;
        # pin it so this branch really exercises the DSL kernel.
        monkeypatch.setattr(flashinfer.norm, "_use_cuda_norm", lambda _device: False)

    batch_size, hidden_size = 8, 1024
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device="cuda")
    gamma = torch.randn(hidden_size, dtype=torch.float32, device="cuda")
    beta = torch.randn(hidden_size, dtype=torch.float32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)

    with pytest.raises(RuntimeError, match="input must be bfloat16"):
        flashinfer.norm.layernorm_quant(out, x.to(torch.float16), gamma, beta, scale)

    x_noncontig = torch.randn(
        batch_size, hidden_size * 2, dtype=torch.bfloat16, device="cuda"
    )[:, :hidden_size]
    with pytest.raises(RuntimeError, match="input must be contiguous"):
        flashinfer.norm.layernorm_quant(out, x_noncontig, gamma, beta, scale)

    with pytest.raises(ValueError, match="scale must be a scalar tensor"):
        flashinfer.norm.layernorm_quant(
            out, x, gamma, beta, torch.ones(2, dtype=torch.float32, device="cuda")
        )


# =============================================================================
# Regression tests for int32 stride overflow
# =============================================================================
# These tests verify that rmsnorm kernels accept tensors with strides exceeding
# INT32_MAX.  The 2D tests use M=2 so that is_contiguous() returns False and the
# non-contiguous kernel path (which uses sym_int64 strides) is exercised.  This
# requires a ~4 GB flat buffer so the large stride is actually traversable.
# The 3D qknorm test can use batch=1 because qk_rmsnorm_cute always uses
# symbolic strides regardless of contiguity.

_INT64_STRIDE = 2**31  # just above INT32_MAX = 2**31 - 1
_STRIDE_BUF_BYTES = (_INT64_STRIDE + 128) * 2  # bf16, H=128


def _skip_if_low_vram():
    free, _ = torch.cuda.mem_get_info()
    if free < _STRIDE_BUF_BYTES * 1.2:
        pytest.skip(
            f"Requires ~{_STRIDE_BUF_BYTES / 1024**3:.1f}GB free VRAM, "
            f"only {free / 1024**3:.1f}GB available"
        )


def test_rmsnorm_int64_stride():
    """2D rmsnorm with row stride > INT32_MAX (issue #3005)."""
    _skip_if_low_vram()
    H = 128
    dtype = torch.bfloat16
    buf = torch.randn(_INT64_STRIDE + H, dtype=dtype, device="cuda")
    w = torch.randn(H, dtype=dtype, device="cuda")

    x = torch.as_strided(buf, (2, H), (_INT64_STRIDE, 1))
    assert not x.is_contiguous()
    y = flashinfer.norm.rmsnorm(x, w)

    y_ref = llama_rms_norm(x.contiguous(), w)
    torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-3)


def test_qknorm_int64_stride():
    """3D qk_rmsnorm with batch stride > INT32_MAX (issue #3005).

    qk_rmsnorm_cute always uses symbolic strides (no contiguity check),
    so batch=1 suffices — the large stride is validated by TVM-FFI even
    though it is never traversed.
    """
    num_heads, head_dim = 4, 128
    dtype = torch.bfloat16
    buf = torch.randn(1, num_heads, head_dim, dtype=dtype, device="cuda")
    w = torch.randn(head_dim, dtype=dtype, device="cuda")

    x = torch.as_strided(buf, (1, num_heads, head_dim), (_INT64_STRIDE, head_dim, 1))
    y = flashinfer.norm.rmsnorm(x, w)

    y_ref = llama_rms_norm(buf, w)
    torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-3)


def test_rmsnorm_quant_int64_stride():
    """rmsnorm_quant with row stride > INT32_MAX (issue #3005)."""
    _skip_if_low_vram()
    H = 128
    dtype = torch.bfloat16
    quant_scale = 1.0
    buf = torch.randn(_INT64_STRIDE + H, dtype=dtype, device="cuda")
    w = torch.randn(H, dtype=dtype, device="cuda")

    x = torch.as_strided(buf, (2, H), (_INT64_STRIDE, 1))
    assert not x.is_contiguous()
    y = torch.empty(2, H, dtype=torch.float8_e4m3fn, device="cuda")
    flashinfer.norm.rmsnorm_quant(y, x, w, torch.tensor(quant_scale, device="cuda"))

    y_ref = llama_rms_norm_quant(x.contiguous(), w, quant_scale, torch.float8_e4m3fn)
    torch.testing.assert_close(y.float(), y_ref.float(), rtol=1, atol=1)


def test_fused_add_rmsnorm_int64_stride():
    """fused_add_rmsnorm with row stride > INT32_MAX (issue #3005)."""
    _skip_if_low_vram()
    H = 128
    dtype = torch.bfloat16
    eps = 1e-6
    buf_x = torch.randn(_INT64_STRIDE + H, dtype=dtype, device="cuda")
    w = torch.randn(H, dtype=dtype, device="cuda")
    # Contiguous residual — only one non-contiguous tensor is needed to
    # trigger the non-contiguous kernel path.
    r = torch.randn(2, H, dtype=dtype, device="cuda")

    x = torch.as_strided(buf_x, (2, H), (_INT64_STRIDE, 1))
    assert not x.is_contiguous()
    x_ref, r_ref = fused_add_rms_norm(x.contiguous().clone(), r.clone(), w, eps)

    flashinfer.fused_add_rmsnorm(x, r, w, eps)

    torch.testing.assert_close(x.contiguous(), x_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(r, r_ref, rtol=1e-3, atol=1e-3)


def test_fused_add_rmsnorm_quant_int64_stride():
    """fused_add_rmsnorm_quant with row stride > INT32_MAX (issue #3005)."""
    _skip_if_low_vram()
    H = 128
    dtype = torch.bfloat16
    eps = 1e-6
    quant_scale = 1.0
    buf_x = torch.randn(_INT64_STRIDE + H, dtype=dtype, device="cuda")
    w = torch.randn(H, dtype=dtype, device="cuda")
    r = torch.randn(2, H, dtype=dtype, device="cuda")

    x = torch.as_strided(buf_x, (2, H), (_INT64_STRIDE, 1))
    assert not x.is_contiguous()
    x_ref, r_ref = fused_add_rms_norm_quant(
        x.contiguous().clone(), r.clone(), w, quant_scale, torch.float8_e4m3fn, eps
    )

    y = torch.empty(2, H, dtype=torch.float8_e4m3fn, device="cuda")
    flashinfer.norm.fused_add_rmsnorm_quant(
        y, x, r, w, torch.tensor(quant_scale, device="cuda"), eps
    )

    torch.testing.assert_close(y.float(), x_ref.float(), rtol=1, atol=1)
    torch.testing.assert_close(r, r_ref, rtol=1e-3, atol=1e-3)


# =============================================================================
# Tests: contiguous tensor with M*H > INT32_MAX
# =============================================================================
# Exercise the contiguous (compact) path of the cute-DSL norm kernels with a
# tensor whose flat element count exceeds 2**31. The compact layout bakes the
# row stride in as a constexpr int, so the offset arithmetic row * H is
# computed in int32 and overflows when M * H > 2**31, producing
# cudaErrorIllegalAddress on the first synchronize.

# Shape (175000, 12288) fp16: 174999 * 12288 > 2**31, while a single row's
# offset fits comfortably in int32.
_OVERFLOW_M = 175000
_OVERFLOW_H = 12288
_OVERFLOW_DTYPE = torch.float16
assert (_OVERFLOW_M - 1) * _OVERFLOW_H > 2**31


def _skip_if_low_vram_for_overflow(extra_bytes_per_elem: float = 4.0):
    """Skip when free VRAM cannot hold ~extra_bytes_per_elem * M * H bytes."""
    need = int(extra_bytes_per_elem * _OVERFLOW_M * _OVERFLOW_H)
    free, _ = torch.cuda.mem_get_info()
    if free < need * 1.15:
        pytest.skip(
            f"Requires ~{need / 1024**3:.1f}GB free VRAM, "
            f"only {free / 1024**3:.1f}GB available"
        )


def _spot_check_rows(y_actual: torch.Tensor, x: torch.Tensor, ref_fn, rtol, atol):
    """Compare the kernel output against a per-row reference on a handful of
    rows that bracket the overflow boundary, plus the first and last rows.
    Avoids materializing a full fp32 reference."""
    boundary_row = (2**31) // _OVERFLOW_H  # first row whose flat offset > INT32_MAX
    rows = sorted(
        {0, 1, boundary_row - 1, boundary_row, boundary_row + 1, _OVERFLOW_M - 1}
    )
    for r in rows:
        if r < 0 or r >= _OVERFLOW_M:
            continue
        row_ref = ref_fn(r)
        torch.testing.assert_close(
            y_actual[r : r + 1].float(), row_ref.float(), rtol=rtol, atol=atol
        )


def test_rmsnorm_contiguous_overflow():
    """rmsnorm on a contiguous tensor with M*H > INT32_MAX."""
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=4.0)  # input + output fp16
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    w = torch.randn(_OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    assert x.is_contiguous()

    y = flashinfer.norm.rmsnorm(x, w)
    torch.cuda.synchronize()  # surface any async illegal-address error

    _spot_check_rows(
        y, x, lambda r: llama_rms_norm(x[r : r + 1], w), rtol=1e-3, atol=1e-3
    )


def test_gemma_rmsnorm_contiguous_overflow():
    """gemma_rmsnorm on a contiguous tensor with M*H > INT32_MAX."""
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=4.0)
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    w = torch.randn(_OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    assert x.is_contiguous()

    y = flashinfer.norm.gemma_rmsnorm(x, w)
    torch.cuda.synchronize()

    _spot_check_rows(
        y, x, lambda r: gemma_rms_norm(x[r : r + 1], w), rtol=1e-3, atol=1e-3
    )


def test_rmsnorm_quant_contiguous_overflow():
    """rmsnorm_quant on a contiguous tensor with M*H > INT32_MAX."""
    # input fp16 (2 B) + fp8 output (1 B).
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=3.0)
    quant_scale = 1.0
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    w = torch.randn(_OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    assert x.is_contiguous()

    y = torch.empty(_OVERFLOW_M, _OVERFLOW_H, dtype=torch.float8_e4m3fn, device="cuda")
    flashinfer.norm.rmsnorm_quant(y, x, w, torch.tensor(quant_scale, device="cuda"))
    torch.cuda.synchronize()

    _spot_check_rows(
        y,
        x,
        lambda r: llama_rms_norm_quant(
            x[r : r + 1], w, quant_scale, torch.float8_e4m3fn
        ),
        rtol=1,
        atol=1,
    )


def test_fused_add_rmsnorm_contiguous_overflow():
    """fused_add_rmsnorm on a contiguous tensor with M*H > INT32_MAX."""
    # input + residual, both fp16; output is written in-place over input.
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=4.0)
    eps = 1e-6
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    r = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    w = torch.randn(_OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    assert x.is_contiguous() and r.is_contiguous()

    # Snapshot the spot-check rows before the in-place op mutates them.
    boundary_row = (2**31) // _OVERFLOW_H
    check_rows = sorted(
        {0, 1, boundary_row - 1, boundary_row, boundary_row + 1, _OVERFLOW_M - 1}
    )
    x_snap = {i: x[i : i + 1].clone() for i in check_rows}
    r_snap = {i: r[i : i + 1].clone() for i in check_rows}

    flashinfer.fused_add_rmsnorm(x, r, w, eps)
    torch.cuda.synchronize()

    for i in check_rows:
        x_ref, r_ref = fused_add_rms_norm(x_snap[i], r_snap[i], w, eps)
        torch.testing.assert_close(x[i : i + 1], x_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(r[i : i + 1], r_ref, rtol=1e-3, atol=1e-3)


def test_gemma_fused_add_rmsnorm_contiguous_overflow():
    """gemma_fused_add_rmsnorm on a contiguous tensor with M*H > INT32_MAX."""
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=4.0)
    eps = 1e-6
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    r = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    w = torch.randn(_OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    assert x.is_contiguous() and r.is_contiguous()

    boundary_row = (2**31) // _OVERFLOW_H
    check_rows = sorted(
        {0, 1, boundary_row - 1, boundary_row, boundary_row + 1, _OVERFLOW_M - 1}
    )
    x_snap = {i: x[i : i + 1].clone() for i in check_rows}
    r_snap = {i: r[i : i + 1].clone() for i in check_rows}

    flashinfer.norm.gemma_fused_add_rmsnorm(x, r, w, eps)
    torch.cuda.synchronize()

    for i in check_rows:
        x_ref, r_ref = gemma_fused_add_rms_norm(x_snap[i], r_snap[i], w, eps)
        torch.testing.assert_close(x[i : i + 1], x_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(r[i : i + 1], r_ref, rtol=1e-3, atol=1e-3)


def test_fused_add_rmsnorm_quant_contiguous_overflow():
    """fused_add_rmsnorm_quant on a contiguous tensor with M*H > INT32_MAX."""
    # input fp16 + residual fp16 + fp8 output.
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=5.0)
    eps = 1e-6
    quant_scale = 1.0
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    r = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    w = torch.randn(_OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    assert x.is_contiguous() and r.is_contiguous()

    boundary_row = (2**31) // _OVERFLOW_H
    check_rows = sorted(
        {0, 1, boundary_row - 1, boundary_row, boundary_row + 1, _OVERFLOW_M - 1}
    )
    x_snap = {i: x[i : i + 1].clone() for i in check_rows}
    r_snap = {i: r[i : i + 1].clone() for i in check_rows}

    y = torch.empty(_OVERFLOW_M, _OVERFLOW_H, dtype=torch.float8_e4m3fn, device="cuda")
    flashinfer.norm.fused_add_rmsnorm_quant(
        y, x, r, w, torch.tensor(quant_scale, device="cuda"), eps
    )
    torch.cuda.synchronize()

    for i in check_rows:
        y_ref, r_ref = fused_add_rms_norm_quant(
            x_snap[i], r_snap[i], w, quant_scale, torch.float8_e4m3fn, eps
        )
        torch.testing.assert_close(y[i : i + 1].float(), y_ref.float(), rtol=1, atol=1)
        torch.testing.assert_close(r[i : i + 1], r_ref, rtol=1e-3, atol=1e-3)


def test_layernorm_contiguous_overflow():
    """layernorm on a contiguous tensor with M*H > INT32_MAX.

    layernorm_cute uses 32-bit symbolic M and row strides, so it overflows on
    the contiguous path the same way rmsnorm does.
    """
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=4.0)
    eps = 1e-6
    dtype = torch.bfloat16  # layernorm requires bf16 input
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=dtype, device="cuda")
    gamma = torch.randn(_OVERFLOW_H, dtype=torch.float32, device="cuda")
    beta = torch.randn(_OVERFLOW_H, dtype=torch.float32, device="cuda")
    assert x.is_contiguous()

    y = flashinfer.layernorm(x, gamma, beta, eps)
    torch.cuda.synchronize()

    boundary_row = (2**31) // _OVERFLOW_H
    rows = sorted(
        {0, 1, boundary_row - 1, boundary_row, boundary_row + 1, _OVERFLOW_M - 1}
    )
    for r in rows:
        ref = F.layer_norm(x[r : r + 1].float(), (_OVERFLOW_H,), gamma, beta, eps).to(
            dtype
        )
        torch.testing.assert_close(y[r : r + 1], ref, rtol=1e-2, atol=1e-2)


def test_qk_rmsnorm_contiguous_overflow():
    """3D qk_rmsnorm with B*N*head_dim > INT32_MAX.

    The 3D path computes the flattened row count M = B*N as int32 inside the
    kernel, so any tensor with B*N*head_dim > 2**31 overflows the row offset.
    """
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=4.0)
    B, N, head_dim = 16800, 1024, 128  # B*N*head_dim = 2_202_009_600 > 2**31
    assert B * N * head_dim > 2**31
    dtype = torch.float16
    x = torch.randn(B, N, head_dim, dtype=dtype, device="cuda")
    w = torch.randn(head_dim, dtype=dtype, device="cuda")
    assert x.is_contiguous()

    y = flashinfer.norm.rmsnorm(x, w)
    torch.cuda.synchronize()

    # Spot-check rows around the boundary in the flattened (B*N, head_dim) view.
    boundary_flat = (2**31) // head_dim
    flat_rows = sorted(
        {0, 1, boundary_flat - 1, boundary_flat, boundary_flat + 1, B * N - 1}
    )
    x_flat = x.view(B * N, head_dim)
    y_flat = y.view(B * N, head_dim)
    for r in flat_rows:
        ref = llama_rms_norm(x_flat[r : r + 1], w)
        torch.testing.assert_close(y_flat[r : r + 1], ref, rtol=1e-3, atol=1e-3)


def test_norm_compilation_without_fp8():
    """Test that norm module compiles successfully without ENABLE_FP8 flag.

    This test verifies the fix for issue #2271 where batchWarpReduceSum in
    reduceKernelUtils.cuh depends on PackType which is only defined when
    ENABLE_FP8 is set. The fix guards batchWarpReduceSum with #ifdef ENABLE_FP8.
    """
    # Create a JIT spec for norm module without ENABLE_FP8 flag
    nvcc_flags = [
        "-DENABLE_BF16",
        # Note: ENABLE_FP8 is intentionally omitted to test compilation without it
    ]
    spec = gen_jit_spec(
        "norm_without_fp8_test",
        [
            jit_env.FLASHINFER_CSRC_DIR / "norm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_norm_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )

    # This should compile successfully without errors
    # If batchWarpReduceSum is not properly guarded, this will fail with:
    # "error: incomplete type is not allowed" for PackType
    module = spec.build_and_load()

    # Verify the module loaded successfully
    assert module is not None


# The public API takes CuTe DSL kernels where they are supported, so call the CUDA kernels.
def _rmsnorm_quant_cuda(out, x, residual, w, scale, enable_pdl):
    flashinfer.norm.get_norm_module().rmsnorm_quant(out, x, w, scale, 1e-6, enable_pdl)


def _fused_add_rmsnorm_quant_cuda(out, x, residual, w, scale, enable_pdl):
    flashinfer.norm.get_norm_module().fused_add_rmsnorm_quant(
        out, x, residual, w, scale, 1e-6, enable_pdl
    )


@pytest.mark.parametrize(
    "norm_quant",
    [
        _rmsnorm_quant_cuda,
        _fused_add_rmsnorm_quant_cuda,
    ],
)
def test_norm_quant_pdl_waits_for_scale(norm_quant):
    x = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
    if not device_support_pdl(x.device):
        pytest.skip("PDL not supported on this device")
    residual = torch.randn_like(x)
    w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    scale_value = 0.5

    expected = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([scale_value], device="cuda")
    norm_quant(expected, x.clone(), residual.clone(), w, scale, enable_pdl=False)

    # The scale the previous kernel is still producing, as in a dynamic-scale pipeline.
    # No other kernel may run between that producer and the norm kernel.
    out = torch.empty_like(expected)
    x, residual = x.clone(), residual.clone()
    scale = torch.tensor([scale_value * 4], device="cuda")
    late_store(scale, scale_value)
    norm_quant(out, x, residual, w, scale, enable_pdl=True)

    torch.testing.assert_close(
        out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )


if __name__ == "__main__":
    # test_norm(1, 1024, torch.float16, False, True, True)
    test_norm(19, 1024, torch.float16, False, True, False)
    # test_fused_add_rmsnorm(1, 16384, torch.float16, True, True)
