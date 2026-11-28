# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""On the device: the locality domain an SM reports is the die whose memory it
reads faster.

One CTA per SM chases a random chain of dependent L2-missing loads through memory
of die 0 and through memory of die 1 (one :func:`flashinfer.cake_locality.
split_empty` allocation) and times each chase with ``%globaltimer``. Each SM
must read the memory of the domain in its ``%locality_domain_id`` register
faster than the other die's memory, by a margin, and the SMs of each domain must
match :func:`flashinfer.cake_locality.sm_table`. On sm_107a boards with 106/106
and 100/112 SMs per die, at 3186 and 4752 MHz memory clocks, the per-SM ratio
of the other die's load latency to the own die's was 1.47-1.57 (305-334 ns own,
471-500 ns other); the test asserts at least :data:`MIN_REMOTE_OVER_LOCAL`.
"""

from __future__ import annotations

import statistics

import pytest
import torch

from flashinfer import cake_locality as cl

# Per SM, median remote load latency over median local load latency.
MIN_REMOTE_OVER_LOCAL = 1.2
# Dependent loads per chase, launches per chase order, attempts per launch to
# place one CTA on every SM.
STEPS = 1024
LAUNCHES = 3
ATTEMPTS = 4
# One chain node every STRIDE_WORDS 4-byte words (256 bytes).
STRIDE_WORDS = 64

_CHASE_SOURCE = r"""
#include <cstdint>

#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

constexpr int kStrideWords = 64;

__device__ __forceinline__ unsigned long long GlobalTimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Thread 0 of CTA c stores (smid, locality domain, supplemental flag, ns of the
// chase through die 0's half, ns through die 1's half) at row c. order 1 chases
// die 1's half first.
__global__ void __launch_bounds__(32, 1)
    ChaseKernel(const unsigned* __restrict__ half0, const unsigned* __restrict__ half1,
                const int* __restrict__ starts, int steps, int order, long long* out) {
  if (threadIdx.x != 0) return;
  unsigned smid, domain, supplemental;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  asm volatile("mov.u32 %0, %%locality_domain_id;" : "=r"(domain));
  asm volatile("mov.u32 %0, %%locality_domain_supplemental_sm;" : "=r"(supplemental));
  long long ns[2];
  unsigned sink = 0;
  for (int i = 0; i < 2; ++i) {
    int side = order ? 1 - i : i;
    const unsigned* p = side ? half1 : half0;
    unsigned index = static_cast<unsigned>(starts[blockIdx.x]);
    unsigned long long t0 = GlobalTimer();
    for (int s = 0; s < steps; ++s) {
      index = __ldcg(p + static_cast<size_t>(index) * kStrideWords);
    }
    sink ^= index;
    ns[side] = static_cast<long long>(GlobalTimer() - t0);
  }
  long long* row = out + static_cast<size_t>(blockIdx.x) * 6;
  row[0] = smid;
  row[1] = domain;
  row[2] = supplemental;
  row[3] = ns[0];
  row[4] = ns[1];
  row[5] = sink;
}

void Chase(tvm::ffi::TensorView chain, int64_t half_words, tvm::ffi::TensorView starts,
           int64_t steps, int64_t order, tvm::ffi::TensorView out) {
  TVM_FFI_ICHECK(chain.ndim() == 1 && chain.size(0) == 2 * half_words);
  TVM_FFI_ICHECK(starts.ndim() == 1 && out.ndim() == 2 && out.size(1) == 6);
  TVM_FFI_ICHECK(out.size(0) == starts.size(0));
  DLDevice device = chain.device();
  TVM_FFI_ICHECK(cudaSetDevice(device.device_id) == cudaSuccess);
  int smem = 0;
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                        device.device_id) == cudaSuccess);
  // Each CTA asks for an SM's whole opt-in shared memory, so no two share an SM.
  TVM_FFI_ICHECK(cudaFuncSetAttribute(ChaseKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem) == cudaSuccess);
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
  const unsigned* base = reinterpret_cast<const unsigned*>(
      static_cast<const char*>(chain.data_ptr()) + chain.byte_offset());
  ChaseKernel<<<static_cast<unsigned>(starts.size(0)), 32, static_cast<size_t>(smem), stream>>>(
      base, base + half_words,
      reinterpret_cast<const int*>(static_cast<const char*>(starts.data_ptr()) +
                                   starts.byte_offset()),
      static_cast<int>(steps), static_cast<int>(order),
      reinterpret_cast<long long*>(static_cast<char*>(out.data_ptr()) + out.byte_offset()));
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(chase, Chase);
"""


def _require_table() -> tuple[int, cl.SmTable]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device) != (10, 7):
        pytest.skip("Cake locality domains need sm_107a")
    from flashinfer.jit.cpp_ext import get_cuda_version

    version = get_cuda_version()
    if (version.major, version.minor) < cl.LOCALITY_CUDA:
        pytest.skip(f"Cake locality domains need a CUDA 13.5 toolkit, found {version}")
    table = cl.sm_table(device)
    if table is None:
        pytest.skip("the device has no Cake SM table")
    return device, table


def _module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec, sm107a_nvcc_flags
    from flashinfer.jit.utils import write_if_different

    name = "test_cake_locality_latency_chase"
    source = jit_env.FLASHINFER_GEN_SRC_DIR / name / "chase.cu"
    source.parent.mkdir(parents=True, exist_ok=True)
    write_if_different(source, _CHASE_SOURCE)
    spec = gen_jit_spec(
        name, [source], extra_cuda_cflags=sm107a_nvcc_flags, use_fast_math=False
    )
    return spec.build_and_load()


def test_locality_domain_is_the_lower_latency_die():
    device, table = _require_table()
    module = _module()
    properties = torch.cuda.get_device_properties(device)
    sm_count = properties.multi_processor_count
    l2_bytes = int(properties.L2_cache_size)
    unit = cl.granularity(device)
    # Each half spans twice the L2 and every CTA chases its own segment of it.
    half_bytes = -(-2 * l2_bytes // unit) * unit
    half_words = half_bytes // 4
    nodes = half_words // STRIDE_WORDS
    assert nodes // sm_count > STEPS
    chain = cl.split_empty((2 * half_words,), torch.int32, half_bytes, device)
    halves = chain.view(2, nodes, STRIDE_WORDS)
    generator = torch.Generator(device=f"cuda:{device}").manual_seed(1)
    order = torch.randperm(
        nodes, generator=generator, device=f"cuda:{device}", dtype=torch.int64
    )
    # The same random cycle over all nodes in both halves: node order[i] links to
    # node order[i + 1].
    halves[:, order, 0] = order.roll(-1).to(torch.int32)
    starts = order[torch.arange(sm_count, device=order.device) * nodes // sm_count]
    starts = starts.to(torch.int32).contiguous()
    flush = torch.empty(2 * l2_bytes, dtype=torch.uint8, device=f"cuda:{device}")
    out = torch.empty((sm_count, 6), dtype=torch.int64, device=f"cuda:{device}")

    ns = {}  # smid -> ([ns of die 0], [ns of die 1])
    domain_of = {}
    for launch in range(2 * LAUNCHES):
        for attempt in range(ATTEMPTS):
            flush.fill_(launch + attempt)  # evict the chains from the L2
            module.chase(chain, half_words, starts, STEPS, launch % 2, out)
            rows = out.tolist()
            if len({row[0] for row in rows}) == sm_count:
                break
        else:
            pytest.fail(f"launch {launch} put no CTA on some SM in {ATTEMPTS} tries")
        for smid, domain, _supplemental, ns0, ns1, _sink in rows:
            assert domain in (0, 1), (smid, domain)
            assert domain_of.setdefault(smid, domain) == domain, smid
            times = ns.setdefault(smid, ([], []))
            times[0].append(ns0)
            times[1].append(ns1)
    torch.cuda.synchronize(device)
    del chain, halves

    assert len(ns) == sm_count
    counts = [0, 0]
    ratios = {}
    for smid, (die0, die1) in ns.items():
        domain = domain_of[smid]
        counts[domain] += 1
        local = statistics.median((die0, die1)[domain])
        remote = statistics.median((die0, die1)[1 - domain])
        ratios[smid] = remote / local
        word = int(table.entries[smid])
        mask = (1 << cl.ENTRY_DOMAIN_BITS) - 1
        table_domain = (word >> cl.ENTRY_DOMAIN_SHIFT) & mask
        assert table_domain == domain, smid
    print(
        f"LOCALITY_LATENCY device={device} counts={counts} table={table.counts} "
        f"remote_over_local min={min(ratios.values()):.3f} "
        f"max={max(ratios.values()):.3f}"
    )
    assert tuple(counts) == tuple(table.counts)
    slow = {
        smid: round(ratio, 3)
        for smid, ratio in ratios.items()
        if ratio < MIN_REMOTE_OVER_LOCAL
    }
    assert not slow, f"SMs whose own die is not clearly nearer: {slow}"
