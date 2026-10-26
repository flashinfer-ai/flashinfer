"""Warp-decode routing reads after a producer that triggers its dependents early.

The warp-decode kernels launch with programmatic dependent launch. A routing
kernel may trigger its dependents before its ``topk_ids`` / ``topk_weights``
stores are visible, and the chain's first kernel reads those inputs before it
could wait on the previous grid. The producer below widens that window on
purpose: it triggers the dependent launch first, spins, and only then copies
the real routing over a decoy routing. A chain that reads the routing early
sees the decoy, so its output differs from the synchronized reference.

The rows cover the three kinds of chain head the generated manifests select on
SM100 and SM103: direct FC1 (reads the routing after its wait), a route-pack
kernel and an FC1 that packs the routes itself (both read the routing first).
The producer and the chain run on one stream with no host synchronization in
between, eagerly and under CUDA Graph replay.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from flashinfer.jit import env as jit_env
from flashinfer.jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags
from flashinfer.jit.utils import write_if_different
from tests.moe.cake_warp_decode_reference import (
    TARGETS,
    Geometry,
    cake_module,
    mutated_topk_ids,
    nondefault_streams,
    prepare_fixture,
    prepare_launch,
    run_cake,
)

_SEED = 20261026
# Long enough that the chain is submitted and resident well before the producer
# writes the routing (about 1 ms at 2 GHz), short enough to keep the test quick.
_SPIN_CYCLES = 2_000_000
_ITERATIONS = 3

_E256_I512 = Geometry("e256_i512_k8", 2048, 512, 256, 8, (1, 18))
_H3072_I384 = Geometry("h3072_e256_i384_k8", 3072, 384, 256, 8, (16,))
# Chain head on both SM100 and SM103: T1 direct FC1, T18 route-pack kernel,
# H3072 I384 T16 FC1 with fused route packing.
_ROWS = [
    pytest.param(geometry, num_tokens, mode, id=f"{label}-{mode}")
    for geometry, num_tokens, label in (
        (_E256_I512, 1, "direct"),
        (_E256_I512, 18, "route_pack"),
        (_H3072_I384, 16, "fused_route_pack"),
    )
    for mode in ("eager", "graph")
]

_TARGET_FLAGS = {"sm100a": sm100a_nvcc_flags, "sm103a": sm103a_nvcc_flags}
_PRODUCER_SOURCE = r"""
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>

using tvm::ffi::TensorView;

__global__ void TriggerThenCopyKernel(uint8_t* ids_out, const uint8_t* ids_in, int64_t ids_bytes,
                                      uint8_t* weights_out, const uint8_t* weights_in,
                                      int64_t weights_bytes, int64_t spin_cycles) {
  // Let the dependent launch start now, before anything is written.
  asm volatile("griddepcontrol.launch_dependents;");
  const int64_t start = clock64();
  while (clock64() - start < spin_cycles) {
  }
  for (int64_t i = threadIdx.x; i < ids_bytes; i += blockDim.x) ids_out[i] = ids_in[i];
  for (int64_t i = threadIdx.x; i < weights_bytes; i += blockDim.x) {
    weights_out[i] = weights_in[i];
  }
}

int64_t StorageBytes(const TensorView& tensor) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA && tensor.IsContiguous())
      << "expected a contiguous CUDA tensor";
  return tensor.numel() * tensor.dtype().bits * tensor.dtype().lanes / 8;
}

void DelayedCopy(TensorView ids_out, TensorView ids_in, TensorView weights_out,
                 TensorView weights_in, int64_t spin_cycles) {
  const int64_t ids_bytes = StorageBytes(ids_out);
  const int64_t weights_bytes = StorageBytes(weights_out);
  TVM_FFI_ICHECK(ids_bytes == StorageBytes(ids_in) && weights_bytes == StorageBytes(weights_in))
      << "size mismatch";
  const DLDevice device = ids_out.device();
  const auto stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
  TriggerThenCopyKernel<<<1, 128, 0, stream>>>(
      static_cast<uint8_t*>(ids_out.data_ptr()), static_cast<const uint8_t*>(ids_in.data_ptr()),
      ids_bytes, static_cast<uint8_t*>(weights_out.data_ptr()),
      static_cast<const uint8_t*>(weights_in.data_ptr()), weights_bytes, spin_cycles);
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(delayed_copy, DelayedCopy);
"""


def _target() -> str:
    if not torch.cuda.is_available():
        pytest.skip("Cake warp decode PDL ordering tests require a CUDA device")
    capability = torch.cuda.get_device_capability(torch.device("cuda"))
    if capability not in TARGETS:
        pytest.skip(
            "Cake warp decode PDL ordering tests require exact SM100 or SM103, "
            f"got SM{capability[0]}{capability[1]}"
        )
    return TARGETS[capability]


@pytest.fixture(scope="module")
def producer():
    target = _target()
    name = f"test_cake_warp_decode_pdl_producer_{target}"
    source = jit_env.FLASHINFER_GEN_SRC_DIR / name / "producer.cu"
    write_if_different(source, _PRODUCER_SOURCE)
    spec = gen_jit_spec(name, [source], extra_cuda_cflags=_TARGET_FLAGS[target])
    return spec.build_and_load()


@pytest.fixture(scope="module")
def fixtures():
    _target()
    cache = {}
    yield cache
    torch.cuda.synchronize()
    cache.clear()


@pytest.mark.gpu
@pytest.mark.parametrize("geometry,num_tokens,mode", _ROWS)
def test_chain_reads_routing_written_after_an_early_trigger(
    producer, fixtures, geometry, num_tokens, mode
):
    if geometry.name not in fixtures:
        fixtures[geometry.name] = prepare_fixture(geometry, _SEED)
    fixture = fixtures[geometry.name]
    device = fixture.device
    real_ids = fixture.topk_ids[:num_tokens].contiguous()
    real_weights = fixture.topk_weights[:num_tokens].contiguous()
    decoy_ids = mutated_topk_ids(geometry, device)[:num_tokens].contiguous()
    decoy_weights = real_weights.flip(1).contiguous()
    # The chain reads only these two buffers; the producer overwrites them.
    ids = torch.empty_like(real_ids)
    weights = torch.empty_like(real_weights)
    routed = replace(fixture, topk_ids=ids, topk_weights=weights)

    (stream,) = nondefault_streams(device, 1)
    module = cake_module(device)
    output, workspace, receipt = prepare_launch(module, fixture, num_tokens)

    def run_chain() -> None:
        run_cake(module, routed, num_tokens, output, workspace, receipt)

    def run_after_producer() -> None:
        producer.delayed_copy(ids, real_ids, weights, real_weights, _SPIN_CYCLES)
        run_chain()

    try:
        outputs = []
        for routing in ((real_ids, real_weights), (decoy_ids, decoy_weights)):
            with torch.cuda.stream(stream):
                ids.copy_(routing[0])
                weights.copy_(routing[1])
            stream.synchronize()
            with torch.cuda.stream(stream):
                run_chain()
                outputs.append(output.clone())
            stream.synchronize()
        expected, stale = outputs
        assert not torch.equal(stale, expected), (
            "the decoy routing must change the output"
        )

        graph = None
        if mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run_after_producer()
        for _ in range(_ITERATIONS):
            with torch.cuda.stream(stream):
                ids.copy_(decoy_ids)
                weights.copy_(decoy_weights)
                output.fill_(float("nan"))
                if graph is None:
                    run_after_producer()
                else:
                    graph.replay()
                actual = output.clone()
            stream.synchronize()
            assert torch.equal(actual, expected), (
                "the chain read the routing before the producer wrote it"
            )
    finally:
        stream.synchronize()
        module.cake_fused_moe_warp_decode_release_workspace(receipt)
