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
"""

"""Benchmark the numerical modes of the experimental Cake dense projection GEMMs (SM100 / SM103 / SM107).

Times the GLM-5.2 training GEMMs of ``flashinfer.experimental.dense_projection_gemm`` per row and arm:

* ``det``: ``deterministic=True`` (the deterministic package's prepared launch, its production plan);
* ``nondet``: ``deterministic=False`` (this package's planner: a split-K program whose parts reduce-add, or the
  deterministic program for a row it keeps whole -- ``nondet_template`` says which);
* ``nondet_p1``: the control of the reduction lever -- this package's program with one split (``splits=1``), where
  the row's program is registered;
* ``cublas``: ``torch.matmul`` on the same strided views (cuBLAS heuristic; bf16 GEMM + cast for the FP32-output
  gradient rows, SGEMM with TF32 off for the router);
* ``bf16x9`` (router rows): ``cublasSgemm`` in ``CUBLAS_FP32_EMULATED_BF16X9_MATH`` (eager strategy).

Rows: the 11 projection rows, the two batched MLA head projections and the FP32 router (``X[T, 6144] @ W[256,
6144].T``) x {forward, input gradient, weight gradient} x T in {16231, 16172}.  The forward and input-gradient
rows of K1 and the router's input gradient run the deterministic program in both modes (nothing to reduce): their
``nondet`` arm is not timed again (``nondet_is_det``).

Timing: ``flashinfer.testing.bench_gpu_time`` with CUPTI activity tracing and a cold L2 between iterations (the
GPU span of every kernel and memset of an iteration, so the router's ``out.zero_()`` is inside), the
deterministic bench's timer.  ``--groups`` alternating groups of every arm (``det``, ``nondet``, ...) per row; a
row reports the median of the group medians per arm, the paired ratios ``nondet / det`` per group, their median
and a one-sided sign test (the probability of at least that many ``nondet`` wins under a fair coin).

Extras per row: the peak temporary and retained device memory of the prepared launch and of the convenience
path of each mode (``--memory``), the error against an FP64 reference (``--accuracy``), the composite
training step ``2 x forward + input gradient + weight gradient`` of the row's projection (the forward recomputed
under activation checkpointing; ``--composite``), and the ``--resident`` variant: every arm re-timed with CUDA
events while a resident kernel on another stream holds ``--resident-sms`` SMs (one block per SM, spinning
``--resident-ms``), which shows a launch that waits for the device to drain (#7038 / #7548).  No arm adds a host
synchronization.

Usage::

    python benchmarks/bench_cake_dense_projection_gemm.py [--rows indexer_hw_wgrad_bf16_t16231 ...]
        [--ops wgrad] [--T 16231,16172] [--f32-grads] [--no-mla] [--no-router] [--steps 20] [--groups 5]
        [--accuracy] [--memory] [--composite] [--resident] [--json out.json] [--csv out.csv]
"""

import argparse
import csv
import ctypes
import json
import math
import statistics
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.experimental.dense_projection_det_gemm import (  # noqa: E402
    cake_backend as det_backend,
)
from flashinfer.experimental.dense_projection_gemm import cake_backend  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402

# GLM-5.2 projection rows: label -> (K = in features, N = out features).
PROJECTION_ROWS = {
    "o_proj": (16384, 6144),
    "q_a": (6144, 2048),
    "q_b": (2048, 16384),
    "kv_a": (6144, 576),
    "shared_gate_up": (6144, 2048),
    "shared_down": (2048, 6144),
    "dense_gate_up": (6144, 12288),
    "dense_down": (12288, 6144),
    "indexer_q": (2048, 4096),
    "indexer_k": (6144, 128),
    "indexer_hw": (6144, 32),
}
MLA_ROWS = {
    "qabs": dict(
        H=64, D_in=192, D_out=512, act_stride_head=256, weight_layout="in_out"
    ),
    "vproj": dict(
        H=64, D_in=512, D_out=256, act_stride_head=512, weight_layout="out_in"
    ),
}
ROUTER_K, ROUTER_N = 6144, 256
OPS = ("fwd", "dgrad", "wgrad")
CANONICAL_T = (16231, 16172)
SEED = 758
CUBLAS_OP_N, CUBLAS_OP_T = 0, 1
CUBLAS_FP32_EMULATED_BF16X9_MATH = 4  # cublasMath_t (CUDA >= 12.9)
CUBLAS_EMULATION_STRATEGY_EAGER = 2  # cublasEmulationStrategy_t (CUDA >= 13.0)


def row_table(T_values, ops, f32_grads, mla, router):
    """label -> row spec, in the order the rows are reported (the deterministic bench's labels)."""
    rows = {}
    for T in T_values:
        for label in PROJECTION_ROWS:
            for op in ops:
                dtypes = ("bf16",) if op == "fwd" or not f32_grads else ("bf16", "f32")
                for dt in dtypes:
                    rows[f"{label}_{op}_{dt}_t{T}"] = dict(
                        family="proj", row=label, op=op, out_dtype=dt, T=T
                    )
        if mla:
            for label in MLA_ROWS:
                for op in ops:
                    rows[f"mla_{label}_{op}_bf16_t{T}"] = dict(
                        family="mla", row=label, op=op, out_dtype="bf16", T=T
                    )
        if router:
            for op in ops:
                rows[f"router_{op}_t{T}"] = dict(
                    family="router", row="router", op=op, out_dtype="f32", T=T
                )
    return rows


def reduces(spec):
    """Whether the row's ``deterministic=False`` launch can differ from the deterministic one: the K1 weight
    gradients and the router forward / weight gradient (``cake_backend.ROUTES``)."""
    if spec["family"] == "router":
        return spec["op"] in ("fwd", "wgrad")
    return spec["op"] == "wgrad"


# ---------------------------------------------------------------------------
# Inputs (the Cake eval harness distributions, as the deterministic bench)
# ---------------------------------------------------------------------------


def make_inputs(spec, seed, device):
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    fam, op, T = spec["family"], spec["op"], int(spec["T"])
    out_dt = {"bf16": torch.bfloat16, "f32": torch.float32}[spec["out_dtype"]]

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, device=device, generator=g) * scale).to(
            torch.bfloat16
        )

    if fam == "proj":
        K, N = PROJECTION_ROWS[spec["row"]]
        X, W, G = rnd(T, K), rnd(N, K, scale=K**-0.5), rnd(T, N)
        if op == "fwd":
            A, B, out = X, W.t(), torch.empty(T, N, device=device, dtype=out_dt)
        elif op == "dgrad":
            A, B, out = G, W, torch.empty(T, K, device=device, dtype=out_dt)
        else:
            A, B, out = G.t(), X, torch.empty(N, K, device=device, dtype=out_dt)
        return dict(A=A, B=B, out=out, X=X, W=W, G=G, flops=2.0 * T * K * N)
    if fam == "mla":
        r = MLA_ROWS[spec["row"]]
        H, Din, Dout, slot = r["H"], r["D_in"], r["D_out"], r["act_stride_head"]
        act = rnd(T, H, slot)[..., :Din]
        if r["weight_layout"] == "in_out":
            weight = rnd(H, Din, Dout, scale=Din**-0.5)
            w_in_out = weight
        else:
            weight = rnd(H, Dout, Din, scale=Din**-0.5)
            w_in_out = weight.transpose(1, 2)
        d_out = rnd(T, H, Dout)
        if op == "fwd":
            A, B = act.permute(1, 0, 2), w_in_out
            out = torch.empty(T, H, Dout, device=device, dtype=out_dt).permute(1, 0, 2)
        elif op == "dgrad":
            A, B = d_out.permute(1, 0, 2), w_in_out.transpose(1, 2)
            out = torch.empty(T, H, slot, device=device, dtype=out_dt)[
                ..., :Din
            ].permute(1, 0, 2)
        elif r["weight_layout"] == "in_out":
            A, B, out = (
                act.permute(1, 2, 0),
                d_out.permute(1, 0, 2),
                torch.empty_like(weight, dtype=out_dt),
            )
        else:
            A, B, out = (
                d_out.permute(1, 2, 0),
                act.permute(1, 0, 2),
                torch.empty_like(weight, dtype=out_dt),
            )
        return dict(A=A, B=B, out=out, flops=2.0 * T * H * Din * Dout)
    K, N = ROUTER_K, ROUTER_N
    X = torch.randn(T, K, device=device, generator=g)
    W = torch.randn(N, K, device=device, generator=g) * 0.02
    G = torch.randn(T, N, device=device, generator=g) * 1e-3
    if op == "fwd":
        A, B, out = X, W.t(), torch.empty(T, N, device=device)
    elif op == "dgrad":
        A, B, out = G, W, torch.empty(T, K, device=device)
    else:
        A, B, out = G.t(), X, torch.empty(N, K, device=device)
    return dict(A=A, B=B, out=out, X=X, W=W, G=G, flops=2.0 * T * K * N)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def prepare(spec, inp, out, *, deterministic, splits=None):
    """The prepared launch of the row in one mode (``splits``: this package's explicit split count)."""
    A, B = inp["A"], inp["B"]
    if spec["family"] == "router":
        return cake_backend.prepare_router_fp32_gemm(
            A, B, out, splits=splits, deterministic=deterministic
        )
    knobs = {} if splits is None else {"splits": splits}
    if spec["family"] == "proj" and spec["op"] == "wgrad" and deterministic:
        return cake_backend.prepare_projection_wgrad(inp["G"], inp["X"], out)
    if spec["family"] == "proj" and spec["op"] == "wgrad":
        # the deterministic package's orientation of the weight gradient, this package's split-K program
        A, B, transposed = det_backend.wgrad_views(
            inp["G"],
            inp["X"],
            arch=det_backend.arch_for(out.device),
            out_f32=out.dtype == torch.float32,
        )
        return cake_backend.prepare_dense_projection_gemm(
            A, B, out, transposed_out=transposed, deterministic=False, **knobs
        )
    return cake_backend.prepare_dense_projection_gemm(
        A, B, out, deterministic=deterministic, **knobs
    )


def convenience(spec, inp, out, *, deterministic):
    """The convenience entry point of the row in one mode (cached plan, arena workspace)."""
    if spec["family"] == "router":
        return lambda: cake_backend.router_fp32_gemm(
            inp["A"], inp["B"], out, deterministic=deterministic
        )
    if spec["family"] == "proj" and spec["op"] == "wgrad":
        return lambda: cake_backend.projection_wgrad(
            inp["G"], inp["X"], out, deterministic=deterministic
        )
    return lambda: cake_backend.dense_projection_gemm(
        inp["A"], inp["B"], out, deterministic=deterministic
    )


def is_atomic(prepared):
    """Whether a prepared launch runs this package's program (else the deterministic package's)."""
    return isinstance(
        prepared,
        (cake_backend.PreparedAtomicGemm, cake_backend.PreparedAtomicRouterGemm),
    )


def torch_arm(spec, inp, out):
    """``torch.matmul`` on the same views (cuBLAS heuristic): bf16 GEMM + cast for the fp32-output gradient rows,
    SGEMM with TF32 disabled for the router rows."""
    A, B = inp["A"], inp["B"]
    if spec["family"] == "router":
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
    direct = out.dtype == A.dtype and out.is_contiguous()

    def run():
        if direct:
            torch.matmul(A, B, out=out)
        else:
            out.copy_(torch.matmul(A, B))
        return out

    return run


class CublasBf16x9:
    """``cublasSgemm`` on a private handle in ``CUBLAS_FP32_EMULATED_BF16X9_MATH`` (eager strategy), on torch's
    current stream."""

    def __init__(self, soname="libcublas.so.13"):
        c = ctypes
        self.lib = lib = c.CDLL(soname)
        lib.cublasCreate_v2.argtypes = [c.POINTER(c.c_void_p)]
        lib.cublasSetMathMode.argtypes = [c.c_void_p, c.c_int]
        lib.cublasSetStream_v2.argtypes = [c.c_void_p, c.c_void_p]
        lib.cublasGetVersion_v2.argtypes = [c.c_void_p, c.POINTER(c.c_int)]
        lib.cublasSgemm_v2.argtypes = [
            c.c_void_p,
            c.c_int,
            c.c_int,
            c.c_int,
            c.c_int,
            c.c_int,
            c.POINTER(c.c_float),
            c.c_void_p,
            c.c_int,
            c.c_void_p,
            c.c_int,
            c.POINTER(c.c_float),
            c.c_void_p,
            c.c_int,
        ]
        self.handle = c.c_void_p()
        self._check(lib.cublasCreate_v2(c.byref(self.handle)), "cublasCreate")
        self._check(
            lib.cublasSetMathMode(self.handle, CUBLAS_FP32_EMULATED_BF16X9_MATH),
            "cublasSetMathMode",
        )
        self.strategy = "library default"
        if hasattr(lib, "cublasSetEmulationStrategy"):
            lib.cublasSetEmulationStrategy.argtypes = [c.c_void_p, c.c_int]
            self._check(
                lib.cublasSetEmulationStrategy(
                    self.handle, CUBLAS_EMULATION_STRATEGY_EAGER
                ),
                "cublasSetEmulationStrategy",
            )
            self.strategy = "eager"
        version = c.c_int()
        self._check(
            lib.cublasGetVersion_v2(self.handle, c.byref(version)), "cublasGetVersion"
        )
        self.version = int(version.value)
        self._alpha, self._beta = c.c_float(1.0), c.c_float(0.0)

    @staticmethod
    def _check(status, what):
        if status != 0:
            raise RuntimeError(f"{what} failed: cublasStatus_t {status}")

    @staticmethod
    def _op(t):
        if t.stride(1) == 1:
            return CUBLAS_OP_N, int(t.stride(0))
        if t.stride(0) == 1:
            return CUBLAS_OP_T, int(t.stride(1))
        raise ValueError("BF16x9 arm: operands need a unit stride on one dimension")

    def arm(self, A, B, out):
        """``out[M, N] = A[M, K] @ B[K, N]`` (row-major ``out``) as the column-major ``out.T = B.T @ A.T``."""
        M, K = (int(s) for s in A.shape)
        N = int(B.shape[1])
        opb, ldb = self._op(B)
        opa, lda = self._op(A)
        alpha, beta = ctypes.byref(self._alpha), ctypes.byref(self._beta)

        def run():
            stream = torch.cuda.current_stream(A.device).cuda_stream
            self._check(
                self.lib.cublasSetStream_v2(self.handle, ctypes.c_void_p(stream)),
                "cublasSetStream",
            )
            self._check(
                self.lib.cublasSgemm_v2(
                    self.handle, opb, opa, N, M, K, alpha, B.data_ptr(), ldb,
                    A.data_ptr(), lda, beta, out.data_ptr(), int(out.stride(0)),
                ),
                "cublasSgemm",
            )  # fmt: skip
            return out

        return run


# ---------------------------------------------------------------------------
# The resident kernel of the #7038 / #7548 variant
# ---------------------------------------------------------------------------

RESIDENT_SOURCE = r"""
extern "C" __global__ void cake_bench_resident(long long cycles) {
  extern __shared__ unsigned char hold[];  // one block per SM: the dynamic shared memory fills the SM
  long long start = clock64();
  if (threadIdx.x == 0) hold[0] = 0;
  while (clock64() - start < cycles) {
    __nanosleep(1000);
  }
}
"""


class ResidentKernel:
    """A kernel that holds ``sms`` SMs (one block of 128 threads and most of the SM's shared memory each) for
    ``ms`` milliseconds on its own stream; compiled once with NVRTC through ``cuda.bindings``."""

    def __init__(self, device, sms, ms):
        from cuda.bindings import driver, nvrtc

        self.driver = driver
        self.sms = int(sms)
        props = torch.cuda.get_device_properties(device)
        # at least ``ms`` at SM clocks up to 2.2 GHz
        self.cycles = int(ms * 1e-3 * 2.2e9)
        major, minor = torch.cuda.get_device_capability(device)
        prog = self._ok(
            nvrtc.nvrtcCreateProgram(
                RESIDENT_SOURCE.encode(), b"resident.cu", 0, [], []
            )
        )
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        result = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if result[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            size = self._ok(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(
                f"resident kernel: NVRTC failed: {log.decode(errors='replace')}"
            )
        size = self._ok(nvrtc.nvrtcGetCUBINSize(prog))
        cubin = b" " * size
        self._ok(nvrtc.nvrtcGetCUBIN(prog, cubin))
        torch.zeros(1, device=device)  # the primary context, current on this thread
        self.module = self._ok(driver.cuModuleLoadData(cubin))
        self.function = self._ok(
            driver.cuModuleGetFunction(self.module, b"cake_bench_resident")
        )
        self.smem = int(getattr(props, "shared_memory_per_block_optin", 232448)) - 1024
        self._ok(
            driver.cuFuncSetAttribute(
                self.function,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                self.smem,
            )
        )
        self.stream = torch.cuda.Stream(device)

    @staticmethod
    def _ok(result):
        status, *values = result if isinstance(result, tuple) else (result,)
        if int(status) != 0:
            raise RuntimeError(f"resident kernel: CUDA call failed with {status}")
        return values[0] if len(values) == 1 else (values or None)

    def launch(self):
        """Enqueue the resident blocks on the resident stream (returns at once)."""
        import numpy as np

        arg = np.array([self.cycles], dtype=np.int64)
        params = np.array([arg.ctypes.data], dtype=np.uint64)
        self._ok(
            self.driver.cuLaunchKernel(
                self.function, self.sms, 1, 1, 128, 1, 1, self.smem,
                self.driver.CUstream(self.stream.cuda_stream), params.ctypes.data, 0,
            )
        )  # fmt: skip
        self._keep = (arg, params)

    def wait(self):
        self.stream.synchronize()


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def median_ms(fn, steps, cupti=True):
    times = bench_gpu_time(
        fn, dry_run_iters=3, repeat_iters=steps, enable_cupti=cupti, cold_l2_cache=True
    )
    return float(statistics.median(times))


def sign_test_p(wins, n):
    """One-sided sign test: the probability of at least ``wins`` successes of ``n`` fair coin flips."""
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n if n else 1.0


def accuracy(inp, out):
    expected = torch.matmul(inp["A"].double(), inp["B"].double())
    got = out.t() if out.shape != expected.shape else out
    diff = (got.double() - expected).abs()
    record = dict(
        finite=bool(torch.isfinite(got).all()),
        max_abs_err=float(diff.max()),
        rel_fro=float(diff.norm() / max(float(expected.norm()), 1e-300)),
        ref_absmax=float(expected.abs().max()),
    )
    if got.dtype == torch.bfloat16:
        record["bf16_violations"] = int((diff > 1e-2 + 1e-2 * expected.abs()).sum())
    return record


def memory_of(make):
    """``(retained, peak_temporary)`` bytes of an arm: what ``make`` (a prepare, or the first convenience call,
    which plans and grows the arena) keeps allocated, and the peak above that during one more call of the
    callable it returns."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    fn = make()
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - base
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base - retained
    return int(retained), int(max(peak, 0))


def first_call(fn):
    fn()
    return fn


def composite_arms(spec, args, device):
    """``2 x forward + input gradient + weight gradient`` of the row's projection, prepared, in both modes: the
    forward recomputed under activation checkpointing, then the backward."""
    if spec["family"] == "mla":
        return None
    grad_dtype = spec["out_dtype"]
    fwd = dict(
        spec, op="fwd", out_dtype="f32" if spec["family"] == "router" else "bf16"
    )
    dgrad, wgrad = (
        dict(spec, op="dgrad", out_dtype=grad_dtype),
        dict(spec, op="wgrad", out_dtype=grad_dtype),
    )
    # one seed: the same X, W, G
    inputs = [make_inputs(s, SEED, device) for s in (fwd, dgrad, wgrad)]
    arms = {}
    for mode, deterministic in (("det", True), ("nondet", False)):
        launches = [
            prepare(
                s, inp, inp["out"], deterministic=deterministic or not reduces(s)
            ).launch
            for s, inp in zip((fwd, dgrad, wgrad), inputs, strict=True)
        ]
        seq = [launches[0], launches[0], launches[1], launches[2]]

        def run(seq=seq):
            for fn in seq:
                fn()

        arms[mode] = run
    return arms


def measure_row(label, spec, args, emu, resident):
    device = args.device
    inp = make_inputs(spec, SEED, device)
    entry = dict(row=label, spec=spec, flops=inp["flops"], reduces=reduces(spec))
    try:
        outs = {}
        arms = {}
        det_out = inp["out"]
        prepared_det = prepare(spec, inp, det_out, deterministic=True)
        arms["det"] = prepared_det.launch
        entry.update(
            det_template=prepared_det.template, det_grid=list(prepared_det.grid)
        )
        if reduces(spec):
            nd_out = torch.empty_like(det_out)
            prepared_nd = prepare(spec, inp, nd_out, deterministic=False)
            arms["nondet"] = prepared_nd.launch
            outs["nondet"] = nd_out
            entry.update(
                nondet_template=prepared_nd.template,
                nondet_atomic=is_atomic(prepared_nd),
                nondet_splits=getattr(prepared_nd, "splits", None),
                nondet_grid=list(prepared_nd.grid),
            )
            try:
                p1_out = torch.empty_like(det_out)
                prepared_p1 = prepare(spec, inp, p1_out, deterministic=False, splits=1)
                if is_atomic(prepared_p1):
                    arms["nondet_p1"] = prepared_p1.launch
                    outs["nondet_p1"] = p1_out
            except (NotImplementedError, ValueError) as exc:
                entry["nondet_p1_unavailable"] = f"{type(exc).__name__}: {exc}"
        else:
            entry["nondet_is_det"] = True
        cub_out = torch.empty_like(det_out)
        arms["cublas"] = torch_arm(spec, inp, cub_out)
        outs["cublas"] = cub_out
        if spec["family"] == "router" and emu is not None:
            emu_out = torch.empty_like(det_out)
            arms["bf16x9"] = emu.arm(inp["A"], inp["B"], emu_out)
            outs["bf16x9"] = emu_out
        # warm-up launches outside the timed region (JIT load, first-call checks)
        for fn in arms.values():
            fn()
        torch.cuda.synchronize()
        if args.accuracy:
            entry["accuracy"] = {"det": accuracy(inp, det_out)}
            for name, out in outs.items():
                entry["accuracy"][name] = accuracy(inp, out)
            if "nondet" in outs:
                entry["nondet_equals_det"] = bool(torch.equal(outs["nondet"], det_out))
        if args.memory:
            mem = {}
            mem["det_prepared_retained"], mem["det_prepared_peak_temp"] = memory_of(
                lambda: prepare(spec, inp, det_out, deterministic=True).launch
            )
            if reduces(spec):
                mem["nondet_prepared_retained"], mem["nondet_prepared_peak_temp"] = (
                    memory_of(
                        lambda: prepare(
                            spec, inp, outs["nondet"], deterministic=False
                        ).launch
                    )
                )
            for mode, deterministic in (("det", True), ("nondet", False)):
                if mode == "nondet" and not reduces(spec):
                    continue
                cake_backend.clear_convenience_caches()
                fn = convenience(spec, inp, det_out, deterministic=deterministic)
                retained, peak = memory_of(lambda fn=fn: first_call(fn))
                (
                    mem[f"{mode}_convenience_retained"],
                    mem[f"{mode}_convenience_peak_temp"],
                ) = retained, peak
            cake_backend.clear_convenience_caches()
            entry["memory_bytes"] = mem
        groups = {name: [] for name in arms}
        for _ in range(args.groups):
            for name, fn in arms.items():
                groups[name].append(median_ms(fn, args.steps))
        entry["ms_groups"] = groups
        entry["ms"] = {name: float(statistics.median(v)) for name, v in groups.items()}
        if "nondet" in groups:
            ratios = [
                n / d for n, d in zip(groups["nondet"], groups["det"], strict=True)
            ]
            wins = sum(r < 1.0 for r in ratios)
            entry.update(
                nondet_over_det=ratios,
                nondet_over_det_median=float(statistics.median(ratios)),
                nondet_wins=wins,
                sign_test_p=sign_test_p(wins, len(ratios)),
            )
        if "nondet_p1" in groups:
            entry["p1_over_det_median"] = float(
                statistics.median(
                    p / d
                    for p, d in zip(groups["nondet_p1"], groups["det"], strict=True)
                )
            )
        entry["cublas_over_det"] = entry["ms"]["cublas"] / entry["ms"]["det"]
        if args.composite and spec["op"] == "wgrad":
            comp = composite_arms(spec, args, device)
            if comp is not None:
                c = {name: [] for name in comp}
                for _ in range(args.groups):
                    for name, fn in comp.items():
                        c[name].append(median_ms(fn, args.steps))
                entry["composite_ms_groups"] = c
                entry["composite_nondet_over_det_median"] = float(
                    statistics.median(
                        n / d for n, d in zip(c["nondet"], c["det"], strict=True)
                    )
                )
        if resident is not None:
            res = {}
            for name, fn in arms.items():
                resident.launch()
                res[name] = median_ms(fn, args.steps, cupti=False)
                resident.wait()
            entry["resident_ms"] = res
            entry["resident"] = dict(sms=resident.sms, cycles=resident.cycles)
        ms = entry["ms"]
        print(
            f"{label:34s} det {ms['det']:8.4f}"
            + (f"  nondet {ms['nondet']:8.4f} x{entry['nondet_over_det_median']:5.3f} "
               f"wins {entry['nondet_wins']}/{args.groups} [{entry['nondet_template']}]" if "nondet" in ms else "  nondet=det")
            + (f"  p1 {ms['nondet_p1']:8.4f}" if "nondet_p1" in ms else "")
            + f"  cublas {ms['cublas']:8.4f}"
            + (f"  bf16x9 {ms['bf16x9']:8.4f}" if "bf16x9" in ms else ""),
            flush=True,
        )  # fmt: skip
    except Exception as exc:  # a failing row is recorded, not hidden
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()
        print(f"{label:34s} failed: {entry['error']}", flush=True)
    torch.cuda.empty_cache()
    return entry


CSV_FIELDS = (
    "row", "det_template", "nondet_template", "nondet_atomic", "nondet_splits", "det_ms", "nondet_ms",
    "nondet_p1_ms", "cublas_ms", "bf16x9_ms", "nondet_over_det_median", "nondet_wins", "sign_test_p",
    "p1_over_det_median", "cublas_over_det", "composite_nondet_over_det_median", "error",
)  # fmt: skip


def csv_row(entry):
    ms = entry.get("ms", {})
    row = {k: entry.get(k, "") for k in CSV_FIELDS}
    for name in ("det", "nondet", "nondet_p1", "cublas", "bf16x9"):
        row[f"{name}_ms"] = ms.get(name, "")
    return row


def library_versions(emu):
    """The library versions a measurement depends on (D8): torch, its CUDA, cuBLAS (runtime), the driver, CUPTI."""
    versions = dict(torch=torch.__version__, torch_cuda=torch.version.cuda)
    versions["cublas"] = emu.version if emu is not None else None
    try:
        from cuda.bindings import driver

        status, value = driver.cuDriverGetVersion()
        versions["cuda_driver"] = int(value) if int(status) == 0 else None
    except Exception as exc:  # noqa: BLE001
        versions["cuda_driver"] = f"unavailable: {type(exc).__name__}"
    try:
        import cupti

        versions["cupti_python"] = getattr(cupti, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        versions["cupti_python"] = f"unavailable: {type(exc).__name__}"
    return versions


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows", nargs="*", default=None, help="row labels (default: every row)"
    )
    parser.add_argument("--ops", default="fwd,dgrad,wgrad")
    parser.add_argument("--T", default=",".join(str(t) for t in CANONICAL_T))
    parser.add_argument(
        "--f32-grads", action="store_true", help="add the FP32-output gradient rows"
    )
    parser.add_argument("--no-mla", action="store_true")
    parser.add_argument("--no-router", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--groups", type=int, default=5, help="alternating groups of every arm per row"
    )
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument(
        "--memory",
        action="store_true",
        help="peak temporary and retained memory per arm",
    )
    parser.add_argument(
        "--composite",
        action="store_true",
        help="2 x forward + input gradient + weight gradient",
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help="re-time every arm next to a resident kernel",
    )
    parser.add_argument("--resident-sms", type=int, default=16)
    parser.add_argument("--resident-ms", type=float, default=2000.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    ops = tuple(o for o in args.ops.split(",") if o)
    unknown = sorted(set(ops) - set(OPS))
    if unknown:
        parser.error(f"unknown ops {unknown}; choose from {list(OPS)}")
    T_values = tuple(int(t) for t in args.T.split(",") if t)
    rows = row_table(T_values, ops, args.f32_grads, not args.no_mla, not args.no_router)
    if args.rows:
        missing = sorted(set(args.rows) - set(rows))
        if missing:
            parser.error(f"unknown rows {missing}")
        rows = {label: rows[label] for label in args.rows}
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    args.device = device
    try:
        emu = CublasBf16x9()
    except Exception as exc:  # noqa: BLE001 - the BF16x9 arm is optional (CUDA >= 12.9 cuBLAS)
        print(f"BF16x9 arm unavailable: {type(exc).__name__}: {exc}", flush=True)
        emu = None
    resident = None
    if args.resident:
        try:
            resident = ResidentKernel(device, args.resident_sms, args.resident_ms)
        except Exception as exc:  # noqa: BLE001 - the variant needs cuda.bindings
            print(
                f"resident variant unavailable: {type(exc).__name__}: {exc}", flush=True
            )
    results = dict(
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        sm_count=int(torch.cuda.get_device_properties(device).multi_processor_count),
        arch=det_backend.arch_for(device),
        versions=library_versions(emu),
        bf16x9_strategy=getattr(emu, "strategy", None),
        steps=args.steps,
        groups=args.groups,
        timer="flashinfer.testing.bench_gpu_time(enable_cupti=True, cold_l2_cache=True); resident: CUDA events",
        rows=[],
    )
    for label, spec in rows.items():
        results["rows"].append(measure_row(label, spec, args, emu, resident))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for entry in results["rows"]:
                writer.writerow(csv_row(entry))
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
