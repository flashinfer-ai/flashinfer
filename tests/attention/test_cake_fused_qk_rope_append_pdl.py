"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Programmatic dependent launch of the Cake fused QK RMSNorm + NeoX RoPE + paged
KV append kernels (BF16 and FP8) behind the kernel that writes ``qkv``.

With ``enable_pdl=True`` the kernel may launch while the preceding kernel on the
stream still runs; it waits for that kernel before it reads ``qkv`` and before
any store.  Two producers write ``qkv`` right before the call:

* a kernel that triggers its dependents first, spins and only then copies the
  real ``qkv`` over a decoy, so a call that read ``qkv`` early would see the
  decoy;
* the QKV projection GEMM (``torch.matmul`` through cuBLAS).

Every call with ``enable_pdl=True`` must write the same bytes as a call with
``enable_pdl=False`` on the ``qkv`` the producer wrote: eagerly, over repeated
calls whose producer output changes, and under CUDA graph replay with the
producer output changing between replays.  The captured edge from the producer
to the kernel is programmatic exactly where the package takes its pdl build
(SM100 / SM103 on the problems ``PDL_POLICY`` admits); every other call is a
plain launch.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest
import torch

import flashinfer.cake_fused_qk_rope_append as bf16_api
import flashinfer.cake_fused_qk_rope_fp8_append as fp8_api
from flashinfer.cake_fused_qk_rope_append_backend import cake_jit as bf16_loader
from flashinfer.cake_fused_qk_rope_fp8_append_backend import cake_jit as fp8_loader
from flashinfer.jit import env as jit_env
from flashinfer.jit.core import (
    gen_jit_spec,
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)
from flashinfer.jit.utils import write_if_different

HEAD_DIM = 128
# Long enough that the call is submitted and resident well before the producer
# writes ``qkv`` (about half a millisecond at 2 GHz).
_SPIN_CYCLES = 1_000_000
# The producer's output generation of every call: repeated, changing calls.
_EAGER_GENERATIONS = (0, 1, 1, 2)
_GRAPH_GENERATIONS = (3, 0, 3, 1)
_GENERATIONS = 4
_GEMM_K = 512
# CUgraphDependencyType of a captured edge.
_DEFAULT_EDGE = 0
_PROGRAMMATIC_EDGE = 1
_ARCH_FLAGS = {
    "sm_90a": sm90a_nvcc_flags,
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}

_PRODUCER_SOURCE = r"""
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>

using tvm::ffi::TensorView;

__global__ void TriggerThenCopyKernel(uint4* out, const uint4* in, int64_t words,
                                      int64_t spin_cycles) {
  // Let the dependent launch start now, before anything is written.
  asm volatile("griddepcontrol.launch_dependents;");
  const int64_t start = clock64();
  while (clock64() - start < spin_cycles) {
  }
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < words;
       i += int64_t(gridDim.x) * blockDim.x) {
    out[i] = in[i];
  }
}

void DelayedCopy(TensorView out, TensorView in, int64_t spin_cycles) {
  TVM_FFI_ICHECK(out.device().device_type == kDLCUDA && out.IsContiguous() &&
                 in.IsContiguous())
      << "expected contiguous CUDA tensors";
  const int64_t bytes = out.numel() * out.dtype().bits / 8;
  TVM_FFI_ICHECK(bytes == in.numel() * in.dtype().bits / 8 && bytes % 16 == 0)
      << "size mismatch";
  const DLDevice device = out.device();
  int sms = 0;
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                        device.device_id) == cudaSuccess);
  const auto stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
  // One resident CTA per SM: every CTA triggers before the dependent can occupy a slot.
  TriggerThenCopyKernel<<<sms, 256, 0, stream>>>(static_cast<uint4*>(out.data_ptr()),
                                                 static_cast<const uint4*>(in.data_ptr()),
                                                 bytes / 16, spin_cycles);
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(delayed_copy, DelayedCopy);
"""


def _device() -> torch.device:
    return torch.device("cuda:0")


def _arch() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    arch = bf16_loader.arch_for_capability(torch.cuda.get_device_capability(_device()))
    if arch is None:
        pytest.skip("the Cake fused QK RoPE append packages ship no build here")
    return arch


@pytest.fixture(scope="module")
def producer():
    arch = _arch()
    name = f"test_cake_fused_qk_rope_append_pdl_producer_{arch}"
    source = jit_env.FLASHINFER_GEN_SRC_DIR / name / "producer.cu"
    write_if_different(source, _PRODUCER_SOURCE)
    spec = gen_jit_spec(name, [source], extra_cuda_cflags=_ARCH_FLAGS[arch])
    return spec.build_and_load()


# ----------------------------------------------------------------------------------------------
# Problems


@dataclass
class Problem:
    """One call of a package on static buffers: ``call(enable_pdl)`` reads ``qkv``
    and writes ``outputs``; ``reset`` restores the outputs' pre-call bytes."""

    qkv: torch.Tensor
    outputs: list[torch.Tensor]
    pristine: list[torch.Tensor]
    call: Callable[[bool], Any]
    stage: Callable[[bool], str]
    sources: list[torch.Tensor] = field(default_factory=list)

    def reset(self) -> None:
        for out, initial in zip(self.outputs, self.pristine, strict=True):
            out.copy_(initial)

    def snapshot(self) -> list[torch.Tensor]:
        return [out.view(torch.uint8).clone() for out in self.outputs]


def _layout(q_lens, ctx_lens, page_size, generator):
    seq_lens = [c + q for c, q in zip(ctx_lens, q_lens, strict=True)]
    pages = [max(1, (s + page_size - 1) // page_size) for s in seq_lens]
    total = sum(pages) + 2
    perm = torch.randperm(total, generator=generator)
    page_indices = torch.zeros((len(q_lens), max(pages)), dtype=torch.int32)
    cursor = 0
    for b, n in enumerate(pages):
        page_indices[b, :n] = perm[cursor : cursor + n].to(torch.int32)
        cursor += n
    q_indptr = torch.zeros(len(q_lens) + 1, dtype=torch.int32)
    q_indptr[1:] = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), 0)
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float64) / half))
    freqs = torch.outer(torch.arange(max(seq_lens) + 1, dtype=torch.float64), inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=1).to(torch.float32)
    device = _device()
    return dict(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        q_indptr=q_indptr.to(device),
        page_indices=page_indices.to(device),
        cos_sin=cos_sin.to(device),
        total_pages=total,
        rows=int(q_indptr[-1]),
    )


def _sources(rows, width, generator):
    """The producer's ``qkv`` of every generation and the decoy it overwrites."""
    device = _device()
    out = [
        torch.randn((rows, width), generator=generator, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
        for _ in range(_GENERATIONS + 1)
    ]
    return out[:-1], out[-1]


def _bf16_problem(case) -> Problem:
    g = torch.Generator(device="cpu").manual_seed(case.seed)
    hq, hkv = case.heads
    lay = _layout(case.q_lens, case.ctx_lens, case.page_size, g)
    device, rows = _device(), lay["rows"]
    sources, decoy = _sources(rows, (hq + 2 * hkv) * HEAD_DIM, g)
    weight = (torch.rand(HEAD_DIM, generator=g) + 0.5).float().to(device)
    cache_shape = (lay["total_pages"], case.page_size, hkv, HEAD_DIM)
    kc0 = torch.randn(cache_shape, generator=g).to(torch.bfloat16).to(device)
    vc0 = torch.randn(cache_shape, generator=g).to(torch.bfloat16).to(device)
    oq0 = torch.full((rows, hq, HEAD_DIM), 1234.0, dtype=torch.bfloat16, device=device)
    kc, vc, oq = kc0.clone(), vc0.clone(), oq0.clone()
    qkv = decoy.clone()

    def call(enable_pdl: bool) -> Any:
        return bf16_api.cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
            qkv,
            lay["cos_sin"],
            lay["seq_lens"],
            lay["q_indptr"],
            lay["page_indices"],
            kc,
            vc,
            num_q_heads=hq,
            num_kv_heads=hkv,
            qk_norm_policy=2,
            q_norm_weight=weight,
            k_norm_weight=weight,
            out_q=oq,
            enable_pdl=enable_pdl,
        )

    def stage(enable_pdl: bool) -> str:
        return bf16_loader.stage_for(
            hq,
            hkv,
            arch=bf16_loader.device_arch(0),
            num_rows=rows,
            num_requests=len(case.q_lens),
            page_size=case.page_size,
            num_sms=bf16_loader.sm_count(0),
            qk_norm_policy=2,
            enable_pdl=enable_pdl,
        )

    return Problem(qkv, [oq, kc, vc], [oq0, kc0, vc0], call, stage, [*sources, decoy])


def _fp8_problem(case) -> Problem:
    g = torch.Generator(device="cpu").manual_seed(case.seed)
    hq, hkv = case.heads
    lay = _layout(case.q_lens, case.ctx_lens, case.page_size, g)
    device, rows, batch = _device(), lay["rows"], len(case.q_lens)
    sources, decoy = _sources(rows, (hq + 2 * hkv) * HEAD_DIM, g)
    weight = (torch.rand(HEAD_DIM, generator=g) + 0.5).float().to(device)
    cache_shape = (lay["total_pages"], case.page_size, hkv, HEAD_DIM)

    def fp8(shape):
        t = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
        t.view(torch.uint8).copy_(
            torch.randint(0, 0x7F, shape, generator=g, dtype=torch.uint8)
        )
        return t

    kc0, vc0, oq0 = fp8(cache_shape), fp8(cache_shape), fp8((rows, hq, HEAD_DIM))
    max_seqlen = max(case.q_lens) if case.prefill else 0
    if case.prefill:
        qs_shape: tuple[int, ...] = (batch, hq, (max_seqlen + 127) // 128 * 128)
    else:
        qs_shape = (rows, hq)
    qs0 = torch.full(qs_shape, float("nan"), device=device)
    flags0 = torch.full((batch, hkv), 7, dtype=torch.int32, device=device)
    kc, vc, oq, qs, flags = (t.clone() for t in (kc0, vc0, oq0, qs0, flags0))
    k_scale = torch.tensor([0.02], dtype=torch.float32, device=device)
    v_scale = torch.tensor([0.03], dtype=torch.float32, device=device)
    qkv = decoy.clone()

    def call(enable_pdl: bool) -> Any:
        return fp8_api.cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache(
            qkv,
            lay["cos_sin"],
            lay["seq_lens"],
            lay["q_indptr"],
            lay["page_indices"],
            (kc, vc),
            case.prefill,
            k_scale,
            v_scale,
            1,
            max_seqlen=max_seqlen,
            q_norm_weight=weight,
            k_norm_weight=weight,
            qk_norm_policy=2,
            out_q=oq,
            q_scale=qs,
            split_k_flag=flags,
            enable_pdl=enable_pdl,
        )

    def stage(enable_pdl: bool) -> str:
        return fp8_loader.stage_for(
            hq,
            hkv,
            arch=fp8_loader.device_arch(0),
            num_rows=rows,
            num_requests=batch,
            page_size=case.page_size,
            num_sms=fp8_loader.sm_count(0),
            is_prefill=case.prefill,
            max_pages_per_request=int(lay["page_indices"].shape[1]),
            has_out_kv=False,
            quant_policy=1,
            enable_pdl=enable_pdl,
        )

    return Problem(
        qkv,
        [oq, qs, flags, kc, vc],
        [oq0, qs0, flags0, kc0, vc0],
        call,
        stage,
        [*sources, decoy],
    )


@dataclass(frozen=True)
class Case:
    """A problem and whether a call that asks for PDL takes the pdl build on
    SM100 / SM103 (never on SM90)."""

    label: str
    route: str
    heads: tuple[int, int]
    q_lens: tuple[int, ...]
    ctx_lens: tuple[int, ...]
    pdl: dict[str, bool]
    page_size: int = 64
    prefill: bool = False
    seed: int = 0

    def problem(self) -> Problem:
        return _bf16_problem(self) if self.route == "bf16" else _fp8_problem(self)


_ON = {"sm_100a": True, "sm_103a": True}
_OFF = {"sm_100a": False, "sm_103a": False}
CASES = [
    Case("bf16_decode_8x1_b32", "bf16", (8, 1), (1,) * 32, (2048,) * 32, _ON, seed=1),
    Case("bf16_decode_64x8_b32", "bf16", (64, 8), (1,) * 32, (2048,) * 32, _ON, seed=2),
    Case("bf16_prefill_8x1_b4q64", "bf16", (8, 1), (64,) * 4, (500,) * 4, _ON, seed=3),
    Case("bf16_prefill_64x8_q128", "bf16", (64, 8), (128,), (1000,), _ON, seed=4),
    Case(
        "bf16_decode_8x1_b512", "bf16", (8, 1), (1,) * 512, (300,) * 512, _OFF, seed=5
    ),
    Case("bf16_prefill_64x8_q1024", "bf16", (64, 8), (1024,), (1024,), _OFF, seed=6),
    Case("fp8_decode_8x1_b32", "fp8", (8, 1), (1,) * 32, (2048,) * 32, _ON, seed=7),
    Case("fp8_decode_64x8_b32", "fp8", (64, 8), (1,) * 32, (2048,) * 32, _ON, seed=8),
    Case(
        "fp8_prefill_8x1_b8q16",
        "fp8",
        (8, 1),
        (16,) * 8,
        (2048,) * 8,
        _ON,
        prefill=True,
        seed=9,
    ),
    Case(
        "fp8_prefill_64x8_b8q16",
        "fp8",
        (64, 8),
        (16,) * 8,
        (2048,) * 8,
        _ON,
        prefill=True,
        seed=10,
    ),
    Case(
        "fp8_decode_8x1_b160",
        "fp8",
        (8, 1),
        (1,) * 160,
        (300,) * 160,
        {"sm_100a": False, "sm_103a": True},
        seed=11,
    ),
    Case(
        "fp8_decode_64x8_b256", "fp8", (64, 8), (1,) * 256, (300,) * 256, _OFF, seed=12
    ),
    Case(
        "fp8_prefill_8x1_q8192",
        "fp8",
        (8, 1),
        (8192,),
        (0,),
        _ON,
        prefill=True,
        seed=13,
    ),
]
_IDS = [case.label for case in CASES]


def _expects_pdl(case: Case, arch: str) -> bool:
    return case.pdl.get(arch, False)


# ----------------------------------------------------------------------------------------------
# Producers


class _TriggerProducer:
    """Triggers its dependents, spins, then copies ``source`` over ``qkv``."""

    def __init__(self, module, problem: Problem) -> None:
        self.module, self.problem = module, problem
        self.source = torch.empty_like(problem.qkv)

    def load(self, generation: int) -> None:
        self.source.copy_(self.problem.sources[generation])

    def run(self) -> None:
        self.module.delayed_copy(self.problem.qkv, self.source, _SPIN_CYCLES)


class _GemmProducer:
    """The QKV projection GEMM: ``qkv = x @ w`` through cuBLAS."""

    def __init__(self, problem: Problem) -> None:
        g = torch.Generator(device="cpu").manual_seed(4242)
        rows, width = problem.qkv.shape
        self.problem = problem
        scale = _GEMM_K**-0.5
        self.w = (
            (torch.randn((_GEMM_K, width), generator=g) * scale)
            .to(torch.bfloat16)
            .to(_device())
        )
        self.xs = [
            torch.randn((rows, _GEMM_K), generator=g).to(torch.bfloat16).to(_device())
            for _ in range(_GENERATIONS)
        ]
        self.x = torch.empty_like(self.xs[0])

    def load(self, generation: int) -> None:
        self.x.copy_(self.xs[generation])

    def run(self) -> None:
        torch.matmul(self.x, self.w, out=self.problem.qkv)


def _producers(case: Case, module) -> list[tuple[str, Problem, Any]]:
    trigger = case.problem()
    gemm = case.problem()
    return [
        ("trigger", trigger, _TriggerProducer(module, trigger)),
        ("gemm", gemm, _GemmProducer(gemm)),
    ]


def _check_against_plain(problem: Problem, got: list[torch.Tensor], label: str) -> None:
    """``got`` (the PDL call's outputs) against a plain call on the ``qkv`` the
    producer left: the same bytes."""
    torch.cuda.synchronize()
    used = problem.qkv.clone()
    problem.reset()
    problem.qkv.copy_(used)
    problem.call(False)
    torch.cuda.synchronize()
    want = problem.snapshot()
    for index, (mine, ref) in enumerate(zip(got, want, strict=True)):
        assert torch.equal(mine, ref), f"{label}: output {index} differs"


def _prepare(problem: Problem, producer, generation: int) -> None:
    problem.reset()
    problem.qkv.copy_(problem.sources[-1])
    producer.load(generation)


# ----------------------------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_enable_pdl_takes_the_pdl_build_where_measured(case):
    arch = _arch()
    problem = case.problem()
    plain, chosen = problem.stage(False), problem.stage(True)
    assert not plain.endswith("_pdl")
    assert chosen == (plain + "_pdl" if _expects_pdl(case, arch) else plain)


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_eager_calls_after_a_producer_match_the_plain_launch(producer, case):
    _arch()
    for name, problem, source in _producers(case, producer):
        # The decoy changes the output: a call that read qkv before the producer
        # wrote it would not match.
        problem.reset()
        problem.qkv.copy_(problem.sources[-1])
        problem.call(False)
        torch.cuda.synchronize()
        stale = problem.snapshot()
        for generation in _EAGER_GENERATIONS:
            _prepare(problem, source, generation)
            source.run()
            problem.call(True)
            torch.cuda.synchronize()
            got = problem.snapshot()
            assert not torch.equal(problem.qkv, problem.sources[-1]), name
            if name == "trigger":
                assert torch.equal(problem.qkv, problem.sources[generation])
            assert any(
                not torch.equal(mine, ref) for mine, ref in zip(got, stale, strict=True)
            ), f"{case.label} {name}: the decoy and the producer output agree"
            _check_against_plain(problem, got, f"{case.label} {name} g{generation}")


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_graph_replays_after_a_producer_match_the_plain_launch(producer, case):
    _arch()
    for name, problem, source in _producers(case, producer):
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            # warm the module and the allocator on the side stream before capture
            _prepare(problem, source, 0)
            source.run()
            problem.call(True)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            source.run()
            problem.call(True)
        stream.synchronize()
        for generation in _GRAPH_GENERATIONS:
            with torch.cuda.stream(stream):
                _prepare(problem, source, generation)
                graph.replay()
            stream.synchronize()
            got = problem.snapshot()
            if name == "trigger":
                assert torch.equal(problem.qkv, problem.sources[generation])
            _check_against_plain(
                problem, got, f"{case.label} {name} replay g{generation}"
            )
        del graph


class _EdgeData(ctypes.Structure):
    """``CUgraphEdgeData`` of the driver API (CUDA 12.3 and newer)."""

    _fields_ = (
        ("from_port", ctypes.c_uint8),
        ("to_port", ctypes.c_uint8),
        ("type", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 5),
    )


def _graph_edges(graph_handle: int) -> list[tuple[int, int, int]]:
    """``(source node, target node, CUgraphDependencyType)`` of every edge, read
    with ``cuGraphGetEdges_v2`` of the driver library itself."""
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
    return [
        (int(sources[i] or 0), int(targets[i] or 0), int(data[i].type))
        for i in range(count.value)
    ]


def _kernel_graph(graph) -> tuple[list[str], list[tuple[int, int, int]]]:
    """Kernel node names and kernel-to-kernel edges of a captured graph that has
    not been instantiated."""
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
        if checked(driver.cuGraphNodeGetType(node)) != (
            driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
        ):
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


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_the_producer_edge_is_programmatic_exactly_for_the_pdl_build(producer, case):
    arch = _arch()
    pytest.importorskip("cuda.bindings.driver")
    if not hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph"):
        pytest.skip("reading graph edges requires CUDAGraph.raw_cuda_graph")
    problem = case.problem()
    source = _TriggerProducer(producer, problem)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        # load both builds before capture
        for enable_pdl in (False, True):
            _prepare(problem, source, 0)
            source.run()
            problem.call(enable_pdl)
    stream.synchronize()
    for enable_pdl in (False, True):
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=stream):
            source.run()
            problem.call(enable_pdl)
        stream.synchronize()
        names, edges = _kernel_graph(graph)
        del graph
        (copy_node,) = [i for i, n in enumerate(names) if "TriggerThenCopyKernel" in n]
        (edge,) = [(target, kind) for src, target, kind in edges if src == copy_node]
        assert "TriggerThenCopyKernel" not in names[edge[0]], names
        programmatic = enable_pdl and _expects_pdl(case, arch)
        expected = _PROGRAMMATIC_EDGE if programmatic else _DEFAULT_EDGE
        assert edge[1] == expected, (names, edges, enable_pdl)
