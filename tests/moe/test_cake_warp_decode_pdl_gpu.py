"""Warp-decode input reads after a producer that triggers its dependents early.

The warp-decode kernels launch with programmatic dependent launch. The kernel
before the chain may trigger its dependents before its stores to the chain's
inputs are visible. A route-pack kernel, or an FC1 that packs the routes
itself, reads ``topk_ids`` before it could wait on the previous grid, so such a
chain must start in stream order. A direct FC1 waits before it reads any input,
so a direct chain keeps programmatic launch for its first kernel too.

The producer below widens the window on purpose: it triggers the dependent
launch first, spins, and only then copies the real routing and activations over
decoys. A chain that reads an input early sees a decoy, so its output differs
from the synchronized reference. A second test reads the dependency edges of a
captured CUDA Graph to check which first kernels keep programmatic launch.

The rows cover the three kinds of chain head the generated manifests select on
SM100 and SM103: direct FC1, a route-pack kernel and an FC1 that packs the
routes itself. The producer and the chain run on one stream with no host
synchronization in between, eagerly and under CUDA Graph replay.
"""

from __future__ import annotations

import ctypes
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
# writes the inputs (about 1 ms at 2 GHz), short enough to keep the test quick.
_SPIN_CYCLES = 2_000_000
_ITERATIONS = 3

_E256_I512 = Geometry("e256_i512_k8", 2048, 512, 256, 8, (1, 18))
_H3072_I384 = Geometry("h3072_e256_i384_k8", 3072, 384, 256, 8, (16,))
# Chain head on both SM100 and SM103: T1 direct FC1, T18 route-pack kernel,
# H3072 I384 T16 FC1 with fused route packing. Each head is listed with whether
# it keeps programmatic launch and the number of kernels in its chain.
_HEADS = (
    (_E256_I512, 1, "direct", True, 3),
    (_E256_I512, 18, "route_pack", False, 4),
    (_H3072_I384, 16, "fused_route_pack", False, 3),
)
_ROWS = [
    pytest.param(geometry, num_tokens, mode, id=f"{label}-{mode}")
    for geometry, num_tokens, label, _, _ in _HEADS
    for mode in ("eager", "graph")
]
_HEAD_ROWS = [
    pytest.param(geometry, num_tokens, programmatic, launches, id=label)
    for geometry, num_tokens, label, programmatic, launches in _HEADS
]
# The caller-written inputs the producer overwrites.
_INPUTS = ("topk_ids", "topk_weights", "hidden_states_q", "hidden_states_scale")
_ROUTING = ("topk_ids", "topk_weights")
_ACTIVATIONS = ("hidden_states_q", "hidden_states_scale")
# CUgraphDependencyType values of a captured dependency edge.
_DEFAULT_EDGE = 0
_PROGRAMMATIC_EDGE = 1

_TARGET_FLAGS = {"sm100a": sm100a_nvcc_flags, "sm103a": sm103a_nvcc_flags}
_PRODUCER_SOURCE = r"""
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>

using tvm::ffi::TensorView;

struct ByteCopy {
  uint8_t* out;
  const uint8_t* in;
  int64_t bytes;
};

struct ByteCopies {
  ByteCopy copies[4];
};

__global__ void TriggerThenCopyKernel(ByteCopies copies, int64_t spin_cycles) {
  // Let the dependent launch start now, before anything is written.
  asm volatile("griddepcontrol.launch_dependents;");
  const int64_t start = clock64();
  while (clock64() - start < spin_cycles) {
  }
  for (const ByteCopy& copy : copies.copies) {
    for (int64_t i = threadIdx.x; i < copy.bytes; i += blockDim.x) copy.out[i] = copy.in[i];
  }
}

int64_t StorageBytes(const TensorView& tensor) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA && tensor.IsContiguous())
      << "expected a contiguous CUDA tensor";
  return tensor.numel() * tensor.dtype().bits * tensor.dtype().lanes / 8;
}

ByteCopy MakeCopy(const TensorView& out, const TensorView& in) {
  const int64_t bytes = StorageBytes(out);
  TVM_FFI_ICHECK(bytes == StorageBytes(in)) << "size mismatch";
  return {static_cast<uint8_t*>(out.data_ptr()), static_cast<const uint8_t*>(in.data_ptr()), bytes};
}

void DelayedCopy(TensorView ids_out, TensorView ids_in, TensorView weights_out,
                 TensorView weights_in, TensorView hidden_out, TensorView hidden_in,
                 TensorView scale_out, TensorView scale_in, int64_t spin_cycles) {
  const ByteCopies copies{{MakeCopy(ids_out, ids_in), MakeCopy(weights_out, weights_in),
                           MakeCopy(hidden_out, hidden_in), MakeCopy(scale_out, scale_in)}};
  const DLDevice device = ids_out.device();
  const auto stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
  TriggerThenCopyKernel<<<1, 128, 0, stream>>>(copies, spin_cycles);
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


def _fixture(fixtures, geometry):
    if geometry.name not in fixtures:
        fixtures[geometry.name] = prepare_fixture(geometry, _SEED)
    return fixtures[geometry.name]


def _reversed_rows(tensor: torch.Tensor) -> torch.Tensor:
    """The same bytes with each row reversed: a valid decoy for any 1-byte dtype."""
    return tensor.view(torch.uint8).flip(1).view(tensor.dtype).contiguous()


def _inputs(fixture, num_tokens):
    """The real inputs of the first ``num_tokens`` rows and a decoy of each."""
    real = {name: getattr(fixture, name)[:num_tokens].contiguous() for name in _INPUTS}
    decoy = {
        "topk_ids": mutated_topk_ids(fixture.geometry, fixture.device)[
            :num_tokens
        ].contiguous(),
        "topk_weights": real["topk_weights"].flip(1).contiguous(),
        "hidden_states_q": _reversed_rows(real["hidden_states_q"]),
        "hidden_states_scale": _reversed_rows(real["hidden_states_scale"]),
    }
    return real, decoy


@pytest.mark.gpu
@pytest.mark.parametrize("geometry,num_tokens,mode", _ROWS)
def test_chain_reads_inputs_written_after_an_early_trigger(
    producer, fixtures, geometry, num_tokens, mode
):
    fixture = _fixture(fixtures, geometry)
    device = fixture.device
    real, decoy = _inputs(fixture, num_tokens)
    # The chain reads only these buffers; the producer overwrites them.
    buffers = {name: torch.empty_like(real[name]) for name in _INPUTS}
    routed = replace(fixture, **buffers)

    (stream,) = nondefault_streams(device, 1)
    module = cake_module(device)
    output, workspace, receipt = prepare_launch(module, fixture, num_tokens)

    def run_chain() -> None:
        run_cake(module, routed, num_tokens, output, workspace, receipt)

    def run_after_producer() -> None:
        producer.delayed_copy(
            *(tensor for name in _INPUTS for tensor in (buffers[name], real[name])),
            _SPIN_CYCLES,
        )
        run_chain()

    def load(stale: tuple[str, ...]) -> None:
        for name in _INPUTS:
            buffers[name].copy_(decoy[name] if name in stale else real[name])

    try:
        # Reference output, then one output per stale input group.
        outputs = []
        for stale in ((), _ROUTING, _ACTIVATIONS):
            with torch.cuda.stream(stream):
                load(stale)
            stream.synchronize()
            with torch.cuda.stream(stream):
                run_chain()
                outputs.append(output.clone())
            stream.synchronize()
        expected, *stale_outputs = outputs
        for stale, stale_output in zip(
            (_ROUTING, _ACTIVATIONS), stale_outputs, strict=True
        ):
            assert not torch.equal(stale_output, expected), (
                f"decoy {stale} must change the output"
            )

        graph = None
        if mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run_after_producer()
        for _ in range(_ITERATIONS):
            with torch.cuda.stream(stream):
                load(_INPUTS)
                output.fill_(float("nan"))
                if graph is None:
                    run_after_producer()
                else:
                    graph.replay()
                actual = output.clone()
            stream.synchronize()
            assert torch.equal(actual, expected), (
                "the chain read an input before the producer wrote it"
            )
    finally:
        stream.synchronize()
        module.cake_fused_moe_warp_decode_release_workspace(receipt)


class _EdgeData(ctypes.Structure):
    """``CUgraphEdgeData`` of the driver API (CUDA 12.3 and newer)."""

    _fields_ = (
        ("from_port", ctypes.c_uint8),
        ("to_port", ctypes.c_uint8),
        ("type", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 5),
    )


def _graph_edges(graph_handle: int) -> list[tuple[int, int, int]]:
    """``(source node, target node, CUgraphDependencyType)`` of every edge of
    the graph, read with ``cuGraphGetEdges_v2`` of the driver library itself.

    cuda-bindings before 12.9.7 and 13.3.0 return the edge data of
    ``cuGraphGetEdges`` from a buffer they have already freed (cuda-python
    issue #1804), so the edge types they report are whatever that memory holds
    by then; the ctypes call keeps the test independent of the bindings' version.
    """
    driver = ctypes.CDLL("libcuda.so.1")
    get_edges = driver.cuGraphGetEdges_v2
    get_edges.restype = ctypes.c_int
    get_edges.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_EdgeData),
        ctypes.POINTER(ctypes.c_size_t),
    )
    count = ctypes.c_size_t(0)
    assert get_edges(graph_handle, None, None, None, ctypes.byref(count)) == 0
    total = count.value
    sources = (ctypes.c_void_p * total)()
    targets = (ctypes.c_void_p * total)()
    data = (_EdgeData * total)()
    assert get_edges(graph_handle, sources, targets, data, ctypes.byref(count)) == 0
    assert count.value == total
    return [
        (int(sources[i] or 0), int(targets[i] or 0), int(data[i].type))
        for i in range(total)
    ]


def _kernel_graph(graph) -> tuple[list[str], list[tuple[int, int, int]]]:
    """Kernel node names and kernel-to-kernel edges (source, target, type) of
    a captured graph that has not been instantiated."""
    driver = pytest.importorskip("cuda.bindings.driver")

    def checked(result):
        error, *rest = result
        assert error == driver.CUresult.CUDA_SUCCESS, repr(error)
        return rest[0] if len(rest) == 1 else rest

    raw = driver.CUgraph(graph.raw_cuda_graph())
    _, count = checked(driver.cuGraphGetNodes(raw, 0))
    nodes, count = checked(driver.cuGraphGetNodes(raw, count))
    index: dict[int, int] = {}
    names = []
    for node in nodes[:count]:
        kind = checked(driver.cuGraphNodeGetType(node))
        if kind != driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        params = checked(driver.cuGraphKernelNodeGetParams(node))
        func = params.func
        if int(func) == 0 and int(params.kern) != 0:
            func = checked(driver.cuKernelGetFunction(params.kern))
        index[int(node)] = len(names)
        names.append(checked(driver.cuFuncGetName(func)).decode())
    edges = [
        (index[source], index[target], kind)
        for source, target, kind in _graph_edges(int(raw))
        if source in index and target in index
    ]
    return names, edges


@pytest.mark.gpu
@pytest.mark.parametrize("geometry,num_tokens,programmatic,launches", _HEAD_ROWS)
def test_first_kernel_launch_dependency(
    producer, fixtures, geometry, num_tokens, programmatic, launches
):
    pytest.importorskip("cuda.bindings.driver")
    if not hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph"):
        pytest.skip("reading graph edges requires CUDAGraph.raw_cuda_graph")
    fixture = _fixture(fixtures, geometry)
    device = fixture.device
    real, _ = _inputs(fixture, num_tokens)
    buffers = {name: torch.empty_like(real[name]) for name in _INPUTS}
    routed = replace(fixture, **buffers)

    (stream,) = nondefault_streams(device, 1)
    module = cake_module(device)
    output, workspace, receipt = prepare_launch(module, fixture, num_tokens)
    try:
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            producer.delayed_copy(
                *(tensor for name in _INPUTS for tensor in (buffers[name], real[name])),
                _SPIN_CYCLES,
            )
            run_cake(module, routed, num_tokens, output, workspace, receipt)
        stream.synchronize()
        names, edges = _kernel_graph(graph)
        del graph
    finally:
        stream.synchronize()
        module.cake_fused_moe_warp_decode_release_workspace(receipt)

    (copy_node,) = [
        i for i, name in enumerate(names) if "TriggerThenCopyKernel" in name
    ]
    assert len(names) == launches + 1, names
    head_edges = [
        (target, kind) for source, target, kind in edges if source == copy_node
    ]
    assert len(head_edges) == 1, (names, edges)
    head, kind = head_edges[0]
    expected_kind = _PROGRAMMATIC_EDGE if programmatic else _DEFAULT_EDGE
    assert kind == expected_kind, (
        f"{names[head]} follows the producer with edge type {kind}, "
        f"expected {expected_kind}"
    )
    # Every later kernel keeps programmatic launch after the one before it.
    chain_edges = [edge for edge in edges if edge[0] != copy_node]
    assert len(chain_edges) == launches - 1, (names, edges)
    assert all(kind == _PROGRAMMATIC_EDGE for _, _, kind in chain_edges), (
        names,
        edges,
    )
