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

Rows: every row of the deterministic family's benchmark table (flashinfer-ai/flashinfer#5677 and #7043): the
projection GEMMs, the batched MLA head projections and the FP32 router x {forward, input gradient, weight gradient}
x their token counts and output dtypes (294 rows: 124 weight gradients, 12 router rows, 158 forward / input-gradient
rows).  Arms per row:

* ``det``: the deterministic package's prepared launch (``dense_projection_det_gemm``, its production plan).
* ``atomic``: ``deterministic=False`` through ``flashinfer.experimental.dense_projection_gemm`` (prepared): this
  package's program where its planner takes the row, else the deterministic program (``atomic_program`` says
  which).  On a delegated row (every forward and input gradient, the rows the planners leave) the arm shows that
  the dispatch costs nothing.
* ``atomic_p1``: the control of the split lever, on the rows the ``atomic`` arm runs this package's program with
  more than one split: the same program at one split (K1: the planner's tile width, ``splits=1``; router: one unit
  per tile).
* ``cublas``: ``torch.matmul`` / ``torch.mm`` / ``torch.bmm`` (cuBLAS heuristic) on the same views; the FP32-output
  gradient rows with ``out_dtype=torch.float32`` (BF16 inputs, FP32 accumulation; a torch without it falls back to a
  BF16 GEMM + cast, marked ``cublas_note``); the router rows as SGEMM with TF32 off.
* ``cublaslt`` (K1 rows): a bounded cuBLASLt sweep -- the heuristic's top ``--lt-algos`` algorithms for the same
  BF16 inputs, FP32 compute, BF16 or FP32 output and layout, each checked against the ``cublas`` result and timed like
  the other arms once; the arm is the fastest.
* ``bf16x9`` (router rows): ``cublasSgemm`` in ``CUBLAS_FP32_EMULATED_BF16X9_MATH`` (eager strategy), cuBLAS's
  FP32-accurate tensor-core path (TF32 and BF16 do not meet the router's accuracy budget).

The non-Cake SOTA latency of a row is its fastest of ``cublas`` and ``cublaslt`` (K1) or ``bf16x9`` (router; plain
SGEMM reported beside it).

Timing: ``flashinfer.testing.bench_gpu_time`` with CUPTI activity tracing and a cold L2 before every iteration (the
GPU span of every kernel and memset of an iteration: the router's ``out.zero_()`` and BF16x3 split are inside), in two
modes -- ``eager`` (the launch called per iteration) and ``graph`` (the launch captured in a CUDA graph and
replayed).  ``--groups`` paired groups per row (5; 9 with ``--groups 9``), the arm order rotated per group and
reversed every other group; per arm and mode the median of the group medians and the 5th / 95th percentiles of all
iterations.  Per mode the ratios ``atomic / det`` of each group, their median, the wins and a one-sided sign test;
``accept`` (the row ships its program) = every group wins and the median ratio is below 0.99, in eager and in graph.

Options: ``--accuracy`` (each arm against an FP64 reference of the same operands, the family's budgets, the worst of
``--acc-reruns`` launches into a NaN-poisoned output; router forward top-8 agreement), ``--memory`` (retained and peak
temporary device memory of the prepared launch and of the convenience path of each mode), ``--composite`` (the
checkpointed training step ``2 x forward + input gradient + weight gradient`` of a projection or router row, both
modes, both timing modes), ``--resident`` (each arm re-timed with CUDA events while a kernel on another stream holds
``--resident-sms`` SMs for ``--resident-us`` from just before every launch: the slowdown of a launch whose grid assumes
the whole device), ``--host-overhead`` (host time per prepared call), ``--exclusive`` (before and after every row,
``nvidia-smi`` must show no other process on the GPU; a row that sees one waits up to ``--exclusive-wait`` seconds and
is flagged ``contended``).  The JSON header records the device, the library versions (torch, CUDA, driver, cuBLAS,
cuBLASLt, CUPTI), the SHA-256 of both packages' host sources and manifests, and the image identity
(``--image-*`` / ``CAKE_IMAGE_*``).  ``--markdown`` writes the two-baseline table (deterministic Cake, non-Cake SOTA,
this package) with absolute latencies per row and mode.

Usage::

    python benchmarks/bench_cake_dense_projection_gemm.py [--rows indexer_hw_wgrad_bf16_t16231 ...]
        [--families proj,mla,router] [--ops fwd,dgrad,wgrad] [--T 16231] [--only-atomic] [--steps 20]
        [--groups 5] [--modes eager,graph] [--accuracy] [--memory] [--composite] [--resident] [--exclusive]
        [--json out.json] [--csv out.csv] [--markdown out.md]
"""

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.experimental.dense_projection_det_gemm import (  # noqa: E402
    cake_backend as det_backend,
)
from flashinfer.experimental.dense_projection_gemm import cake_backend  # noqa: E402
from flashinfer.testing import bench_gpu_time  # noqa: E402

# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

GLM_T = (16172, 16231)
FOLLOWUP_T = (16231, 16384)
# Projection shapes: name -> (K = in features, N = out features, operations, token counts).  Operation codes: F the
# forward (BF16 out), D / d the input gradient with BF16 / FP32 out, W the weight gradient with BF16 and FP32 out.
# fmt: off
PROJECTION_ROWS = {
    "indexer_q": (2048, 4096, "FDW", GLM_T),
    "shared_down": (2048, 6144, "FDdW", GLM_T),
    "q_b": (2048, 16384, "FDdW", GLM_T),
    "4096to32": (4096, 32, "FDdW", FOLLOWUP_T),
    "4096to48": (4096, 48, "FDdW", FOLLOWUP_T),
    "4096to64": (4096, 64, "FDdW", FOLLOWUP_T),
    "4096to128": (4096, 128, "FDdW", FOLLOWUP_T),
    "4096to512": (4096, 512, "FDdW", FOLLOWUP_T),
    "4096to1024": (4096, 1024, "FDW", FOLLOWUP_T),
    "4096to4096": (4096, 4096, "FDW", FOLLOWUP_T),
    "4096to12288": (4096, 12288, "FDW", FOLLOWUP_T),
    "5120to32": (5120, 32, "FDdW", FOLLOWUP_T),
    "5120to48": (5120, 48, "FDdW", FOLLOWUP_T),
    "5120to64": (5120, 64, "FDdW", FOLLOWUP_T),
    "5120to128": (5120, 128, "FDdW", FOLLOWUP_T),
    "5120to512": (5120, 512, "FDdW", FOLLOWUP_T),
    "5120to1024": (5120, 1024, "FDW", FOLLOWUP_T),
    "5120to5120": (5120, 5120, "FDW", FOLLOWUP_T),
    "5120to17408": (5120, 17408, "FDW", FOLLOWUP_T),
    "indexer_hw": (6144, 32, "FDdW", GLM_T),
    "indexer_k": (6144, 128, "FDdW", GLM_T),
    "kv_a": (6144, 576, "FDdW", GLM_T),
    "q_a": (6144, 2048, "FDdW", GLM_T),
    "shared_gate_up": (6144, 2048, "FDdW", GLM_T),
    "dense_gate_up": (6144, 12288, "W", GLM_T),
    "8192to128": (8192, 128, "FDdW", FOLLOWUP_T),
    "12288to4096": (12288, 4096, "FDW", FOLLOWUP_T),
    "dense_down": (12288, 6144, "W", GLM_T),
    "o_proj": (16384, 6144, "FDdW", GLM_T),
    "17408to5120": (17408, 5120, "FDW", FOLLOWUP_T),
}
# fmt: on
MLA_ROWS = {
    "qabs": dict(
        H=64, D_in=192, D_out=512, act_stride_head=256, weight_layout="in_out"
    ),
    "vproj": dict(
        H=64, D_in=512, D_out=256, act_stride_head=512, weight_layout="out_in"
    ),
}
# Router rows: hidden size -> token counts.
ROUTER_ROWS = {4096: (16231,), 5120: (16231,), 6144: GLM_T}
ROUTER_N = 256
OPS = ("fwd", "dgrad", "wgrad")
FAMILIES = ("proj", "mla", "router")
SEED = 758
ACCEPT_RATIO = 0.99


def row_table(T_override=None):
    """label -> row spec, in the order the rows are reported (the deterministic bench's label scheme)."""
    rows = {}
    for name, (K, N, codes, Ts) in PROJECTION_ROWS.items():
        for T in T_override or Ts:
            for code in codes:
                ops = {
                    "F": [("fwd", "bf16")],
                    "D": [("dgrad", "bf16")],
                    "d": [("dgrad", "f32")],
                    "W": [("wgrad", "bf16"), ("wgrad", "f32")],
                }[code]
                for op, dt in ops:
                    rows[f"{name}_{op}_{dt}_t{T}"] = dict(
                        family="proj", row=name, K=K, N=N, op=op, out_dtype=dt, T=T
                    )
    for name in MLA_ROWS:
        for T in T_override or GLM_T:
            for op in OPS:
                rows[f"mla_{name}_{op}_bf16_t{T}"] = dict(
                    family="mla", row=name, op=op, out_dtype="bf16", T=T
                )
    for K, Ts in ROUTER_ROWS.items():
        for T in T_override or Ts:
            for op in OPS:
                rows[f"router_h{K}_{op}_t{T}"] = dict(
                    family="router", row=f"router_h{K}", K=K, N=ROUTER_N, op=op,
                    out_dtype="f32", T=T,
                )  # fmt: skip
    return rows


def k1_family(spec):
    return spec["family"] in ("proj", "mla")


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
        K, N = spec["K"], spec["N"]
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
    K, N = spec["K"], spec["N"]
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


def fresh_out(out):
    """A separate output with the geometry of ``out`` (the same strides, so the same plan)."""
    storage = torch.empty(
        out.untyped_storage().nbytes() // out.element_size(),
        device=out.device,
        dtype=out.dtype,
    )
    return torch.as_strided(storage, out.shape, out.stride(), out.storage_offset())


# ---------------------------------------------------------------------------
# Cake arms
# ---------------------------------------------------------------------------


def prepare_det(spec, inp, out):
    """The deterministic package's prepared launch of the row (its production plan)."""
    if spec["family"] == "router":
        return det_backend.prepare_router_fp32_det_gemm(inp["A"], inp["B"], out)
    if spec["family"] == "proj" and spec["op"] == "wgrad":
        return det_backend.prepare_projection_det_wgrad(inp["G"], inp["X"], out)
    return det_backend.prepare_dense_projection_det_gemm(inp["A"], inp["B"], out)


def prepare_atomic(spec, inp, out, **knobs):
    """``deterministic=False`` through this package (``knobs``: ``splits`` / ``block_n``, the control arm)."""
    if spec["family"] == "router":
        return cake_backend.prepare_router_fp32_gemm(
            inp["A"], inp["B"], out, deterministic=False, **knobs
        )
    if spec["family"] == "proj" and spec["op"] == "wgrad":
        if not knobs:
            return cake_backend.prepare_projection_wgrad(
                inp["G"], inp["X"], out, deterministic=False
            )
        return cake_backend.prepare_dense_projection_gemm(
            inp["X"].t(),
            inp["G"],
            out,
            transposed_out=True,
            deterministic=False,
            **knobs,
        )
    return cake_backend.prepare_dense_projection_gemm(
        inp["A"], inp["B"], out, deterministic=False, **knobs
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


def plan_record(prepared):
    """The plan facts of a prepared launch (and the rule-table key and value of this package's program, so a
    per-row verdict can be written into ``K1_SPLIT_RULES`` / ``ROUTER_RULES``)."""
    record = dict(template=prepared.template, grid=list(prepared.grid))
    if isinstance(prepared, cake_backend.PreparedAtomicGemm):
        p = prepared.plan
        record.update(
            block_n=p.block_n,
            splits=p.splits,
            tiles=p.tiles,
            swapped=p.swapped,
            ws_bytes=4 * p.ws_f32_elems + 4 * p.counters_u32,
            rule_key=[
                prepared.arch,
                p.L,
                p.M,
                p.N,
                "f32" if p.out_f32 else "bf16",
            ],
            rule_value=[p.block_n, p.splits],
        )
    elif isinstance(prepared, cake_backend.PreparedAtomicRouterGemm):
        p = prepared.plan
        record.update(
            units=p.units,
            head_spans=p.head_spans,
            head_steps=p.head_steps,
            tiles=p.tiles,
            k_iters=p.k_iters,
            part_tiles=p.part_tiles,
            reproducible=p.reproducible,
            rule_key=[
                prepared.arch,
                p.layout,
                p.K if p.layout == "kk" else p.M,
                p.N,
            ],
            rule_value=p.units,
        )
    elif hasattr(prepared, "splits"):
        record["splits"] = prepared.splits
    return record


# ---------------------------------------------------------------------------
# Library arms
# ---------------------------------------------------------------------------


def torch_arm(spec, inp, out):
    """``torch.matmul`` / ``mm`` / ``bmm`` on the same views (cuBLAS heuristic)."""
    A, B = inp["A"], inp["B"]
    note = None
    if spec["family"] == "router":
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
    mm = torch.bmm if A.dim() == 3 else torch.mm
    if out.dtype == torch.float32 and A.dtype == torch.bfloat16:

        def run_f32():
            if out.is_contiguous():
                mm(A, B, out_dtype=torch.float32, out=out)
            else:
                out.copy_(mm(A, B, out_dtype=torch.float32))
            return out

        try:
            run_f32()
            torch.cuda.synchronize()
            return run_f32, note
        except (TypeError, RuntimeError) as exc:
            note = f"bf16 GEMM + cast (no out_dtype: {type(exc).__name__})"
    direct = out.dtype == A.dtype and out.is_contiguous()

    def run():
        if direct:
            torch.matmul(A, B, out=out)
        else:
            out.copy_(torch.matmul(A, B))
        return out

    return run, note


CUBLAS_OP_N, CUBLAS_OP_T = 0, 1
CUBLAS_FP32_EMULATED_BF16X9_MATH = 4  # cublasMath_t (CUDA >= 12.9)
CUBLAS_EMULATION_STRATEGY_EAGER = 2  # cublasEmulationStrategy_t (CUDA >= 13.0)


class CublasBf16x9:
    """``cublasSgemm`` on a private handle in ``CUBLAS_FP32_EMULATED_BF16X9_MATH`` (eager strategy), on torch's
    current stream."""

    def __init__(self, sonames=("libcublas.so.13", "libcublas.so.12")):
        c = ctypes
        self.lib = lib = _cdll(sonames)
        lib.cublasCreate_v2.argtypes = [c.POINTER(c.c_void_p)]
        lib.cublasSetMathMode.argtypes = [c.c_void_p, c.c_int]
        lib.cublasSetStream_v2.argtypes = [c.c_void_p, c.c_void_p]
        lib.cublasGetVersion_v2.argtypes = [c.c_void_p, c.POINTER(c.c_int)]
        lib.cublasSgemm_v2.argtypes = [
            c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int, c.POINTER(c.c_float), c.c_void_p, c.c_int,
            c.c_void_p, c.c_int, c.POINTER(c.c_float), c.c_void_p, c.c_int,
        ]  # fmt: skip
        lib.cublasSetWorkspace_v2.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t]
        self.handle = c.c_void_p()
        _check(lib.cublasCreate_v2(c.byref(self.handle)), "cublasCreate")
        _check(
            lib.cublasSetMathMode(self.handle, CUBLAS_FP32_EMULATED_BF16X9_MATH),
            "cublasSetMathMode",
        )
        # a caller-owned workspace: no allocation inside a captured CUDA graph
        self.workspace = torch.empty(32 << 20, device="cuda", dtype=torch.uint8)
        _check(
            lib.cublasSetWorkspace_v2(
                self.handle, self.workspace.data_ptr(), self.workspace.numel()
            ),
            "cublasSetWorkspace",
        )
        self.strategy = "library default"
        if hasattr(lib, "cublasSetEmulationStrategy"):
            lib.cublasSetEmulationStrategy.argtypes = [c.c_void_p, c.c_int]
            _check(
                lib.cublasSetEmulationStrategy(
                    self.handle, CUBLAS_EMULATION_STRATEGY_EAGER
                ),
                "cublasSetEmulationStrategy",
            )
            self.strategy = "eager"
        version = c.c_int()
        _check(
            lib.cublasGetVersion_v2(self.handle, c.byref(version)), "cublasGetVersion"
        )
        self.version = int(version.value)
        self._alpha, self._beta = c.c_float(1.0), c.c_float(0.0)

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
            _check(
                self.lib.cublasSetStream_v2(self.handle, ctypes.c_void_p(stream)),
                "cublasSetStream",
            )
            _check(
                self.lib.cublasSgemm_v2(
                    self.handle, opb, opa, N, M, K, alpha, B.data_ptr(), ldb,
                    A.data_ptr(), lda, beta, out.data_ptr(), int(out.stride(0)),
                ),
                "cublasSgemm",
            )  # fmt: skip
            return out

        return run


def _cdll(sonames):
    errors = []
    for soname in sonames:
        try:
            return ctypes.CDLL(soname)
        except OSError as exc:
            errors.append(f"{soname}: {exc}")
    raise OSError("; ".join(errors))


def _check(status, what):
    if status != 0:
        raise RuntimeError(f"{what} failed: cublasStatus_t {status}")


# cuBLASLt enums (cublasLt.h, cublas_api.h, library_types.h)
CUBLAS_COMPUTE_32F = 68
CUDA_R_32F, CUDA_R_16BF = 0, 14
CUBLASLT_MATMUL_DESC_TRANSA, CUBLASLT_MATMUL_DESC_TRANSB = 3, 4
CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5
CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6
CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1


class _LtHeuristic(ctypes.Structure):
    _fields_ = [
        ("algo", ctypes.c_uint64 * 8),
        ("workspace_size", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves_count", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


class CublasLt:
    """A bounded cuBLASLt sweep: the heuristic's top algorithms for one BF16 GEMM ``out[L, M, N] = A[L, M, K] @
    B[L, K, N]`` (FP32 compute and scale, BF16 or FP32 output, the views' strides and batch offsets), each a
    callable on torch's current stream."""

    def __init__(
        self, sonames=("libcublasLt.so.13", "libcublasLt.so.12"), workspace_mib=32
    ):
        c = ctypes
        self.lib = lib = _cdll(sonames)
        vp = c.c_void_p
        lib.cublasLtCreate.argtypes = [c.POINTER(vp)]
        lib.cublasLtGetVersion.restype = c.c_size_t
        lib.cublasLtMatmulDescCreate.argtypes = [c.POINTER(vp), c.c_int, c.c_int]
        lib.cublasLtMatmulDescSetAttribute.argtypes = [vp, c.c_int, vp, c.c_size_t]
        lib.cublasLtMatrixLayoutCreate.argtypes = [
            c.POINTER(vp), c.c_int, c.c_uint64, c.c_uint64, c.c_int64,
        ]  # fmt: skip
        lib.cublasLtMatrixLayoutSetAttribute.argtypes = [vp, c.c_int, vp, c.c_size_t]
        lib.cublasLtMatmulPreferenceCreate.argtypes = [c.POINTER(vp)]
        lib.cublasLtMatmulPreferenceSetAttribute.argtypes = [
            vp, c.c_int, vp, c.c_size_t,
        ]  # fmt: skip
        lib.cublasLtMatmulAlgoGetHeuristic.argtypes = [
            vp, vp, vp, vp, vp, vp, vp, c.c_int, c.POINTER(_LtHeuristic), c.POINTER(c.c_int),
        ]  # fmt: skip
        lib.cublasLtMatmul.argtypes = [
            vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, c.c_size_t, vp,
        ]  # fmt: skip
        self.handle = vp()
        _check(lib.cublasLtCreate(c.byref(self.handle)), "cublasLtCreate")
        self.version = int(lib.cublasLtGetVersion())
        self.workspace_bytes = int(workspace_mib) << 20
        self.workspace = None
        self._alpha, self._beta = c.c_float(1.0), c.c_float(0.0)

    def _layout(self, dtype, rows, cols, ld, batch, offset):
        c = ctypes
        layout = c.c_void_p()
        _check(
            self.lib.cublasLtMatrixLayoutCreate(c.byref(layout), dtype, rows, cols, ld),
            "cublasLtMatrixLayoutCreate",
        )
        if batch > 1:
            count, stride = c.c_int32(batch), c.c_int64(offset)
            _check(
                self.lib.cublasLtMatrixLayoutSetAttribute(
                    layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, c.byref(count), 4
                ),
                "BATCH_COUNT",
            )
            _check(
                self.lib.cublasLtMatrixLayoutSetAttribute(
                    layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, c.byref(stride), 8
                ),
                "STRIDED_BATCH_OFFSET",
            )  # fmt: skip
        return layout

    def algos(self, A, B, out, top):
        """``[(index, run), ...]``: callables of the heuristic's first ``top`` algorithms (an empty list when cuBLASLt
        offers none for the layout)."""
        c = ctypes
        A3 = A if A.dim() == 3 else A.unsqueeze(0)
        B3 = B if B.dim() == 3 else B.unsqueeze(0)
        O3 = out if out.dim() == 3 else out.unsqueeze(0)
        L, M, K = (int(s) for s in A3.shape)
        N = int(B3.shape[2])
        if O3.stride(2) != 1:
            raise ValueError("cuBLASLt arm: out needs a unit stride on n")
        if self.workspace is None:
            self.workspace = torch.empty(
                self.workspace_bytes, device=A.device, dtype=torch.uint8
            )
        # column-major D = out^T [N, M] = op(B^T) [N, K] @ op(A^T) [K, M]
        if B3.stride(2) == 1:
            op_a, a_layout = CUBLAS_OP_N, (N, K, B3.stride(1))
        elif B3.stride(1) == 1:
            op_a, a_layout = CUBLAS_OP_T, (K, N, B3.stride(2))
        else:
            raise ValueError("cuBLASLt arm: B needs a unit stride on k or n")
        if A3.stride(2) == 1:
            op_b, b_layout = CUBLAS_OP_N, (K, M, A3.stride(1))
        elif A3.stride(1) == 1:
            op_b, b_layout = CUBLAS_OP_T, (M, K, A3.stride(2))
        else:
            raise ValueError("cuBLASLt arm: A needs a unit stride on m or k")
        out_type = CUDA_R_32F if out.dtype == torch.float32 else CUDA_R_16BF
        desc = c.c_void_p()
        _check(
            self.lib.cublasLtMatmulDescCreate(
                c.byref(desc), CUBLAS_COMPUTE_32F, CUDA_R_32F
            ),
            "cublasLtMatmulDescCreate",
        )
        for attr, value in (
            (CUBLASLT_MATMUL_DESC_TRANSA, op_a),
            (CUBLASLT_MATMUL_DESC_TRANSB, op_b),
        ):
            v = c.c_int32(value)
            _check(
                self.lib.cublasLtMatmulDescSetAttribute(desc, attr, c.byref(v), 4),
                "cublasLtMatmulDescSetAttribute",
            )
        la = self._layout(CUDA_R_16BF, *a_layout, L, B3.stride(0))
        lb = self._layout(CUDA_R_16BF, *b_layout, L, A3.stride(0))
        ld = self._layout(out_type, N, M, O3.stride(1), L, O3.stride(0))
        pref = c.c_void_p()
        _check(
            self.lib.cublasLtMatmulPreferenceCreate(c.byref(pref)),
            "cublasLtMatmulPreferenceCreate",
        )
        ws = c.c_uint64(self.workspace_bytes)
        _check(
            self.lib.cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, c.byref(ws), 8
            ),
            "cublasLtMatmulPreferenceSetAttribute",
        )
        results = (_LtHeuristic * int(top))()
        found = c.c_int(0)
        status = self.lib.cublasLtMatmulAlgoGetHeuristic(
            self.handle, desc, la, lb, ld, ld, pref, int(top), results, c.byref(found)
        )
        if status != 0:
            return []
        keep = (desc, la, lb, ld, pref, results)
        runs = []
        for i in range(int(found.value)):
            if results[i].state != 0:
                continue

            def run(i=i, keep=keep):
                stream = torch.cuda.current_stream(A.device).cuda_stream
                _check(
                    self.lib.cublasLtMatmul(
                        self.handle, desc, c.byref(self._alpha), B3.data_ptr(), la,
                        A3.data_ptr(), lb, c.byref(self._beta), O3.data_ptr(), ld,
                        O3.data_ptr(), ld, c.byref(results[i].algo), self.workspace.data_ptr(),
                        self.workspace_bytes, c.c_void_p(stream),
                    ),
                    "cublasLtMatmul",
                )  # fmt: skip
                return out

            runs.append((i, run))
        return runs


# ---------------------------------------------------------------------------
# The resident kernel of the #7038 / #7548 variant
# ---------------------------------------------------------------------------

RESIDENT_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(128) cake_bench_resident(long long cycles) {
  extern __shared__ unsigned char hold[];  // one block per SM: the dynamic shared memory fills the SM
  long long start = clock64();
  if (threadIdx.x == 0) hold[0] = 0;
  while (clock64() - start < cycles) {
    __nanosleep(100);
  }
}
"""


class ResidentKernel:
    """A kernel that holds ``sms`` SMs (one block of 128 threads and most of the SM's shared memory each) for
    ``us`` microseconds on its own stream; compiled once with NVRTC to ``compute_80`` PTX (the driver JIT-compiles
    it for any SM100 / SM103 / SM107 part) through ``cuda.bindings``."""

    def __init__(self, device, sms, us):
        from cuda.bindings import driver, nvrtc

        self.driver = driver
        self.sms = int(sms)
        self.us = float(us)
        props = torch.cuda.get_device_properties(device)
        # at least ``us`` at SM clocks up to 2.2 GHz
        self.cycles = int(us * 1e-6 * 2.2e9)
        prog = self._ok(
            nvrtc.nvrtcCreateProgram(
                RESIDENT_SOURCE.encode(), b"resident.cu", 0, [], []
            )
        )
        opts = [b"--gpu-architecture=compute_80"]
        result = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if self._code(result[0]) != 0:
            size = self._ok(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(
                f"resident kernel: NVRTC failed: {log.decode(errors='replace')}"
            )
        size = self._ok(nvrtc.nvrtcGetPTXSize(prog))
        ptx = b" " * size
        self._ok(nvrtc.nvrtcGetPTX(prog, ptx))
        torch.zeros(1, device=device)  # the primary context, current on this thread
        self.module = self._ok(driver.cuModuleLoadData(ptx))
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
    def _code(status):
        return int(getattr(status, "value", status))

    @classmethod
    def _ok(cls, result):
        status, *values = result if isinstance(result, tuple) else (result,)
        if cls._code(status) != 0:
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


def resident_ms(fn, resident, steps, flush):
    """Median CUDA-event time of ``fn`` launched right after the resident kernel (each iteration: flush the L2,
    wait for it, start the resident blocks, launch ``fn`` between two events on the current stream)."""
    times = []
    stream = torch.cuda.current_stream()
    for i in range(steps + 2):
        flush.zero_()
        resident.stream.wait_stream(stream)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        resident.launch()
        start.record(stream)
        fn()
        end.record(stream)
        stream.wait_stream(resident.stream)
        end.synchronize()
        if i >= 2:
            times.append(start.elapsed_time(end))
    return float(statistics.median(times))


def event_ms(fn, steps, flush):
    """Median CUDA-event time of ``fn`` after an L2 flush (the resident variant's timer without the resident)."""
    times = []
    stream = torch.cuda.current_stream()
    for i in range(steps + 2):
        flush.zero_()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record(stream)
        fn()
        end.record(stream)
        end.synchronize()
        if i >= 2:
            times.append(start.elapsed_time(end))
    return float(statistics.median(times))


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def timed(fn, steps, graph):
    """Per-iteration GPU times (ms): CUPTI span, cold L2, eager or captured in a CUDA graph."""
    return [
        float(t)
        for t in bench_gpu_time(
            fn,
            dry_run_iters=3,
            repeat_iters=steps,
            enable_cupti=True,
            cold_l2_cache=True,
            use_cuda_graph=graph,
        )
    ]


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def sign_test_p(wins, n):
    """One-sided sign test: the probability of at least ``wins`` successes of ``n`` fair coin flips."""
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n if n else 1.0


def paired_groups(arms, groups, steps, modes):
    """``{mode: {arm: [group medians]}}`` and ``{mode: {arm: [all iterations]}}``: per group every arm in a rotated
    order (reversed every other group), each mode back to back."""
    names = list(arms)
    medians = {m: {n: [] for n in names} for m in modes}
    samples = {m: {n: [] for n in names} for m in modes}
    errors = {}
    for g in range(groups):
        order = names[g % len(names) :] + names[: g % len(names)]
        if g % 2:
            order = order[::-1]
        for name in order:
            for mode in modes:
                if (mode, name) in errors:
                    continue
                try:
                    times = timed(arms[name], steps, mode == "graph")
                except Exception as exc:  # noqa: BLE001 - an arm a mode cannot run is recorded
                    errors[(mode, name)] = f"{type(exc).__name__}: {exc}"[:300]
                    continue
                medians[mode][name].append(float(statistics.median(times)))
                samples[mode][name].extend(times)
    return medians, samples, errors


def summarize(medians, samples, errors):
    """Per mode: per arm the median of the group medians and p5 / p95; ``atomic`` against ``det`` and
    ``atomic_p1`` against ``det``: per-group ratios, median, wins, sign test, accept."""
    out = {}
    for mode, per_arm in medians.items():
        rec = dict(ms={}, p5_ms={}, p95_ms={}, groups_ms={}, errors={})
        for name, values in per_arm.items():
            if (mode, name) in errors:
                rec["errors"][name] = errors[(mode, name)]
                continue
            rec["ms"][name] = float(statistics.median(values))
            rec["p5_ms"][name] = percentile(samples[mode][name], 0.05)
            rec["p95_ms"][name] = percentile(samples[mode][name], 0.95)
            rec["groups_ms"][name] = values
        for arm in ("atomic", "atomic_p1"):
            if arm in rec["ms"] and "det" in rec["ms"]:
                ratios = [
                    a / d for a, d in zip(per_arm[arm], per_arm["det"], strict=True)
                ]
                wins = sum(r < 1.0 for r in ratios)
                rec[f"{arm}_over_det"] = dict(
                    ratios=ratios,
                    median=float(statistics.median(ratios)),
                    wins=wins,
                    groups=len(ratios),
                    sign_test_p=sign_test_p(wins, len(ratios)),
                    accept=wins == len(ratios) and statistics.median(ratios) < ACCEPT_RATIO,
                )  # fmt: skip
        out[mode] = rec
    return out


def sota_choice(spec, ms):
    """The non-Cake SOTA arm of a row: the router's BF16x9 (the FP32-accurate class; plain SGEMM is reported beside
    it), else the fastest of the cuBLAS heuristic and the cuBLASLt sweep."""
    if spec["family"] == "router":
        return "bf16x9" if "bf16x9" in ms else None
    present = [n for n in ("cublas", "cublaslt") if n in ms]
    return min(present, key=lambda n: ms[n]) if present else None


def accuracy(spec, inp, out, ref=None):
    """The error of ``out`` against FP64 of the same operands and the family's budget."""
    expected = (
        ref if ref is not None else torch.matmul(inp["A"].double(), inp["B"].double())
    )
    got = out
    diff = (got.double() - expected).abs()
    ref_absmax = float(expected.abs().max())
    record = dict(
        finite=bool(torch.isfinite(got).all()),
        max_abs_err=float(diff.max()),
        rel_fro=float(diff.norm() / max(float(expected.norm()), 1e-300)),
        ref_absmax=ref_absmax,
    )
    if spec["family"] == "router":
        record["pass"] = record["finite"] and record["rel_fro"] <= 2e-6
        if spec["op"] == "fwd":
            top = torch.topk(got.double(), 8, dim=1).indices.sort(dim=1).values
            ref_top = torch.topk(expected, 8, dim=1).indices.sort(dim=1).values
            record["top8_rows_differ"] = int((top != ref_top).any(dim=1).sum())
            record["pass"] = record["pass"] and record["top8_rows_differ"] == 0
    elif got.dtype == torch.bfloat16:
        record["bf16_violations"] = int((diff > 1e-2 + 1e-2 * expected.abs()).sum())
        record["pass"] = record["finite"] and record["bf16_violations"] == 0
    else:
        record["pass"] = (
            record["finite"]
            and record["rel_fro"] <= 5e-5
            and record["max_abs_err"] <= 1e-3 * ref_absmax
        )
    return record


def worst_accuracy(spec, inp, fn, out, reruns, ref):
    worst = None
    for _ in range(max(1, reruns)):
        out.fill_(float("nan"))
        fn()
        torch.cuda.synchronize()
        rec = accuracy(spec, inp, out, ref)
        if worst is None:
            worst = rec
        else:
            for k in ("max_abs_err", "rel_fro", "bf16_violations", "top8_rows_differ"):
                if k in rec:
                    worst[k] = max(worst[k], rec[k])
            worst["pass"] = worst["pass"] and rec["pass"]
            worst["finite"] = worst["finite"] and rec["finite"]
    worst["reruns"] = max(1, reruns)
    return worst


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


def host_us_per_call(fn, steps):
    """Mean host (CPU) time per call in microseconds over ``steps`` back-to-back calls (not GPU time)."""
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    host = time.perf_counter() - start
    torch.cuda.synchronize()
    return host / steps * 1e6


def lt_sweep(lt, spec, inp, reference, top, steps):
    """The fastest correct cuBLASLt algorithm of the row (eager CUPTI median, one pass): ``(run, record, out)``
    (``run`` None when no algorithm qualifies; ``record`` says why)."""
    out = fresh_out(inp["out"])
    record = dict(algos=[], top=top)
    best = None
    try:
        runs = lt.algos(inp["A"], inp["B"], out, top)
    except Exception as exc:  # noqa: BLE001 - a layout cuBLASLt does not take is recorded
        record["unsupported"] = f"{type(exc).__name__}: {exc}"[:300]
        return None, record, None
    if not runs:
        record["unsupported"] = "no heuristic algorithm"
    tol = 2e-2 if out.dtype == torch.bfloat16 else 1e-3
    for index, run in runs:
        entry = dict(index=index)
        try:
            out.fill_(float("nan"))
            run()
            torch.cuda.synchronize()
            err = float(
                (out.double() - reference.double()).norm()
                / max(float(reference.double().norm()), 1e-300)
            )
            entry["rel_vs_cublas"] = err
            if not math.isfinite(err) or err > tol:
                entry["rejected"] = "result differs from torch.matmul"
            else:
                entry["ms"] = float(statistics.median(timed(run, steps, False)))
                if best is None or entry["ms"] < best[1]:
                    best = (run, entry["ms"], index)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
        record["algos"].append(entry)
    if best is None:
        record.setdefault("unsupported", "no correct algorithm")
        return None, record, out
    record["best_index"] = best[2]
    return best[0], record, out


def build_arms(spec, inp, args, libs, entry):
    """The arms of a row (name -> callable writing its own output) and their outputs."""
    arms, outs, prepared = {}, {}, {}
    det_out = fresh_out(inp["out"])
    p = prepare_det(spec, inp, det_out)
    arms["det"], outs["det"], prepared["det"] = p.launch, det_out, p
    entry["det_plan"] = plan_record(p)
    at_out = fresh_out(inp["out"])
    p = prepare_atomic(spec, inp, at_out)
    arms["atomic"], outs["atomic"], prepared["atomic"] = p.launch, at_out, p
    entry["atomic_program"] = is_atomic(p)
    entry["atomic_plan"] = plan_record(p)
    control = None
    if isinstance(p, cake_backend.PreparedAtomicGemm) and p.splits > 1:
        control = dict(splits=1, block_n=p.plan.block_n)
    elif (
        isinstance(p, cake_backend.PreparedAtomicRouterGemm)
        and p.plan.units != p.plan.tiles
    ):
        control = dict(splits=1)
    if control is not None and not args.no_p1:
        p1_out = fresh_out(inp["out"])
        try:
            q = prepare_atomic(spec, inp, p1_out, **control)
            if is_atomic(q):
                arms["atomic_p1"], outs["atomic_p1"], prepared["atomic_p1"] = (
                    q.launch,
                    p1_out,
                    q,
                )
                entry["atomic_p1_plan"] = plan_record(q)
        except (NotImplementedError, ValueError) as exc:
            entry["atomic_p1_unavailable"] = f"{type(exc).__name__}: {exc}"
    cub_out = fresh_out(inp["out"])
    run, note = torch_arm(spec, inp, cub_out)
    arms["cublas"], outs["cublas"] = run, cub_out
    if note:
        entry["cublas_note"] = note
    if spec["family"] == "router" and libs.get("bf16x9") is not None:
        emu_out = fresh_out(inp["out"])
        try:
            arms["bf16x9"] = libs["bf16x9"].arm(inp["A"], inp["B"], emu_out)
            outs["bf16x9"] = emu_out
        except ValueError as exc:
            entry["bf16x9_unsupported"] = str(exc)
    elif spec["family"] == "router":
        entry["bf16x9_unsupported"] = libs.get("bf16x9_error", "unavailable")
    return arms, outs, prepared


def exclusive_check(args, uuid):
    """Other processes on this GPU (``nvidia-smi --query-compute-apps``), excluding this one."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader", "-i", uuid],
            capture_output=True, text=True, timeout=60,
        )  # fmt: skip
    except Exception as exc:  # noqa: BLE001
        return [f"nvidia-smi unavailable: {type(exc).__name__}"]
    me = str(os.getpid())
    return [
        line.strip()
        for line in res.stdout.splitlines()
        if line.strip() and line.split(",")[0].strip() != me
    ]


def wait_exclusive(args, uuid):
    deadline = time.time() + args.exclusive_wait
    others = exclusive_check(args, uuid)
    while others and time.time() < deadline:
        time.sleep(5)
        others = exclusive_check(args, uuid)
    return others


def composite(spec, args, libs):
    """``2 x forward + input gradient + weight gradient`` of the row's projection or router (the forward recomputed
    under activation checkpointing; prepared launches): ``det``, ``atomic`` and the non-Cake sequence (``cublas``:
    ``torch.matmul`` per operation; the router: ``bf16x9`` per operation)."""
    if spec["family"] == "mla" or spec["op"] != "wgrad":
        return None
    device = args.device
    fwd_dt = "f32" if spec["family"] == "router" else "bf16"
    steps = [
        dict(spec, op="fwd", out_dtype=fwd_dt),
        dict(spec, op="dgrad", out_dtype=fwd_dt),
        dict(spec, op="wgrad"),
    ]
    inputs = [make_inputs(s, SEED, device) for s in steps]
    library = (
        "bf16x9"
        if spec["family"] == "router" and libs.get("bf16x9") is not None
        else "cublas"
    )
    arms = {}
    for mode in ("det", "atomic", library):
        launches = []
        for s, inp in zip(steps, inputs, strict=True):
            o = fresh_out(inp["out"])
            if mode == "det":
                launches.append(prepare_det(s, inp, o).launch)
            elif mode == "atomic":
                launches.append(prepare_atomic(s, inp, o).launch)
            elif mode == "bf16x9":
                launches.append(libs["bf16x9"].arm(inp["A"], inp["B"], o))
            else:
                launches.append(torch_arm(s, inp, o)[0])
        seq = [launches[0], launches[0], launches[1], launches[2]]

        def run(seq=seq):
            for fn in seq:
                fn()

        arms[mode] = run
    return arms


def measure_row(label, spec, args, libs, resident, uuid):
    entry = dict(row=label, spec=spec)
    try:
        _measure_row(entry, spec, args, libs, resident, uuid)
    except Exception as exc:  # a failing row is recorded, not hidden
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()
        print(f"{label:34s} failed: {entry['error']}", flush=True)
    finally:
        cake_backend.clear_convenience_caches()
        torch.cuda.empty_cache()
    return entry


def _add_sota(spec, modes):
    for rec in modes.values():
        choice = sota_choice(spec, rec["ms"])
        rec["sota_arm"], rec["sota_ms"] = choice, rec["ms"].get(choice)


def _measure_row(entry, spec, args, libs, resident, uuid):
    device = args.device
    if args.exclusive:
        entry["foreign_before"] = wait_exclusive(args, uuid)
    inp = make_inputs(spec, SEED, device)
    entry["flops"] = inp["flops"]
    arms, outs, prepared = build_arms(spec, inp, args, libs, entry)
    if args.only_atomic and not entry["atomic_program"]:
        entry["skipped"] = "the atomic arm runs the deterministic program"
        return
    # warm-up launches outside the timed region (JIT load, first-call checks)
    for fn in arms.values():
        fn()
    torch.cuda.synchronize()
    if k1_family(spec) and libs.get("lt") is not None and not args.no_lt:
        run, record, lt_out = lt_sweep(
            libs["lt"], spec, inp, outs["cublas"], args.lt_algos, args.steps
        )
        entry["cublaslt_sweep"] = record
        if run is not None:
            arms["cublaslt"], outs["cublaslt"] = run, lt_out
    elif k1_family(spec):
        entry["cublaslt_sweep"] = dict(
            unsupported=libs.get("lt_error", "skipped (--no-lt)")
        )
    if args.accuracy:
        ref = torch.matmul(inp["A"].double(), inp["B"].double())
        acc = {}
        for name, fn in arms.items():
            reruns = args.acc_reruns if name.startswith("atomic") else 1
            acc[name] = worst_accuracy(spec, inp, fn, outs[name], reruns, ref)
        entry["accuracy"] = acc
        entry["atomic_equals_det"] = bool(torch.equal(outs["atomic"], outs["det"]))
        del ref
    if args.memory:
        mem = {}
        mem["det_prepared_retained"], mem["det_prepared_peak_temp"] = memory_of(
            lambda: prepare_det(spec, inp, outs["det"]).launch
        )
        mem["atomic_prepared_retained"], mem["atomic_prepared_peak_temp"] = memory_of(
            lambda: prepare_atomic(spec, inp, outs["atomic"]).launch
        )
        for mode, deterministic in (("det", True), ("atomic", False)):
            cake_backend.clear_convenience_caches()
            fn = convenience(spec, inp, outs[mode], deterministic=deterministic)
            retained, peak = memory_of(lambda fn=fn: first_call(fn))
            mem[f"{mode}_convenience_retained"] = retained
            mem[f"{mode}_convenience_peak_temp"] = peak
        cake_backend.clear_convenience_caches()
        entry["memory_bytes"] = mem
    if args.host_overhead:
        entry["host_us_prepared"] = {
            name: host_us_per_call(prepared[name].launch, args.steps)
            for name in prepared
        }
    medians, samples, errors = paired_groups(arms, args.groups, args.steps, args.modes)
    entry["modes"] = summarize(medians, samples, errors)
    _add_sota(spec, entry["modes"])
    verdicts = [
        entry["modes"][m].get("atomic_over_det", {}).get("accept", False)
        for m in args.modes
    ]
    entry["accept"] = bool(entry["atomic_program"] and all(verdicts))
    if args.composite:
        comp = composite(spec, args, libs)
        if comp is not None:
            entry["composite"] = summarize(
                *paired_groups(comp, args.groups, args.steps, args.modes)
            )
            _add_sota(spec, entry["composite"])
            del comp
    if resident is not None:
        flush = torch.empty(
            int(det_backend.device_facts(device).l2_bytes) * 2 // 4,
            device=device,
            dtype=torch.float32,
        )
        res = {}
        for name in [n for n in ("det", "atomic", "atomic_p1") if n in arms]:
            plain = event_ms(arms[name], args.steps, flush)
            with_resident = resident_ms(arms[name], resident, args.steps, flush)
            res[name] = dict(
                events_ms=plain,
                resident_ms=with_resident,
                slowdown=with_resident / plain,
            )
        res["atomic_over_det"] = (
            res["atomic"]["resident_ms"] / res["det"]["resident_ms"]
        )
        entry["resident"] = dict(
            sms=resident.sms, us=resident.us, cycles=resident.cycles, arms=res
        )
        del flush
    if args.exclusive:
        entry["foreign_after"] = exclusive_check(args, uuid)
        entry["contended"] = bool(entry.get("foreign_before") or entry["foreign_after"])
    line = f"{entry['row']:34s}"
    for mode in args.modes:
        rec = entry["modes"][mode]
        ms = rec["ms"]
        r = rec.get("atomic_over_det", {})
        line += (
            f" | {mode} det {ms.get('det', float('nan')) * 1e3:8.1f}us "
            f"atomic {ms.get('atomic', float('nan')) * 1e3:8.1f}us "
            f"x{r.get('median', float('nan')):.3f} {r.get('wins', 0)}/{r.get('groups', 0)}"
            + (f" sota[{rec['sota_arm']}] {rec['sota_ms'] * 1e3:8.1f}us" if rec.get("sota_ms") else "")
        )  # fmt: skip
    line += (
        f" | {'atomic' if entry['atomic_program'] else 'det'} {entry['atomic_plan']['template']}"
        + (" ACCEPT" if entry["accept"] else "")
    )
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

CSV_FIELDS = (
    "row", "mode", "atomic_program", "accept", "det_template", "atomic_template", "atomic_splits", "atomic_units",
    "det_ms", "atomic_ms", "atomic_p1_ms", "cublas_ms", "cublaslt_ms", "bf16x9_ms", "sota_arm", "sota_ms",
    "det_p5_ms", "det_p95_ms", "atomic_p5_ms", "atomic_p95_ms", "atomic_over_det_median", "atomic_wins", "groups",
    "sign_test_p", "atomic_p1_over_det_median", "det_over_atomic", "sota_over_atomic", "contended", "error",
)  # fmt: skip


def csv_rows(entry, modes):
    for mode in modes:
        rec = entry.get("modes", {}).get(mode, {})
        ms = rec.get("ms", {})
        r = rec.get("atomic_over_det", {})
        row = dict(
            row=entry["row"],
            mode=mode,
            atomic_program=entry.get("atomic_program", ""),
            accept=entry.get("accept", ""),
            det_template=entry.get("det_plan", {}).get("template", ""),
            atomic_template=entry.get("atomic_plan", {}).get("template", ""),
            atomic_splits=entry.get("atomic_plan", {}).get("splits", ""),
            atomic_units=entry.get("atomic_plan", {}).get("units", ""),
            sota_arm=rec.get("sota_arm", ""),
            sota_ms=rec.get("sota_ms", ""),
            atomic_over_det_median=r.get("median", ""),
            atomic_wins=r.get("wins", ""),
            groups=r.get("groups", ""),
            sign_test_p=r.get("sign_test_p", ""),
            atomic_p1_over_det_median=rec.get("atomic_p1_over_det", {}).get(
                "median", ""
            ),
            contended=entry.get("contended", ""),
            error=entry.get("error", ""),
        )
        for name in ("det", "atomic", "atomic_p1", "cublas", "cublaslt", "bf16x9"):
            row[f"{name}_ms"] = ms.get(name, "")
        for name in ("det", "atomic"):
            row[f"{name}_p5_ms"] = rec.get("p5_ms", {}).get(name, "")
            row[f"{name}_p95_ms"] = rec.get("p95_ms", {}).get(name, "")
        if ms.get("atomic"):
            row["det_over_atomic"] = ms["det"] / ms["atomic"] if ms.get("det") else ""
            row["sota_over_atomic"] = (
                rec["sota_ms"] / ms["atomic"] if rec.get("sota_ms") else ""
            )
        yield row


def _us(rec, name, reason=None):
    ms = rec.get("ms", {}).get(name)
    if ms is None:
        why = rec.get("errors", {}).get(name) or reason
        return f"unsupported ({why})" if why else "unsupported"
    return f"{ms * 1e3:.1f} us [{rec['p5_ms'][name] * 1e3:.1f}, {rec['p95_ms'][name] * 1e3:.1f}]"


def _shape(spec):
    if spec["family"] == "mla":
        r = MLA_ROWS[spec["row"]]
        return (
            f"MLA {spec['row']} (H={r['H']}, {r['D_in']}->{r['D_out']}) T={spec['T']}"
        )
    if spec["family"] == "router":
        return f"router {spec['K']}->{spec['N']} T={spec['T']}"
    named = "" if spec["row"][0].isdigit() else f"{spec['row']} "
    return f"{named}{spec['K']}->{spec['N']} T={spec['T']}"


def _md_line(gpu, workload, operation, rec, reason):
    ms = rec["ms"]
    sota = rec.get("sota_arm")
    sota_cell = f"{_us(rec, sota)} ({sota})" if sota else _us(rec, "", reason)
    det_x = ms["det"] / ms["atomic"] if "det" in ms and "atomic" in ms else None
    sota_x = rec["sota_ms"] / ms["atomic"] if sota and "atomic" in ms else None
    return (
        f"| {gpu} | {workload} | {operation} | {_us(rec, 'det')} | {sota_cell} | {_us(rec, 'atomic')} | "
        f"{'' if det_x is None else f'{det_x:.3f}'} | {'' if sota_x is None else f'{sota_x:.3f}'} |"
    )


def markdown(results, modes):
    """The two-baseline table: one line per row, timed operation and timing mode (latency = median of the group
    medians, [p5, p95] of all iterations), then the caption."""
    gpu = f"{results['device']} / {results['arch']}"
    lines = [
        "| GPU / architecture | workload / shape / dtype / mode | operation | deterministic Cake latency | "
        "non-Cake SOTA latency | new non-deterministic Cake latency | speedup vs deterministic Cake | "
        "speedup vs non-Cake SOTA |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in results["rows"]:
        if "modes" not in entry:
            continue
        spec = entry["spec"]
        program = "" if entry["atomic_program"] else ", runs the deterministic program"
        reason = entry.get("bf16x9_unsupported") or entry.get("cublaslt_sweep", {}).get(
            "unsupported"
        )
        for mode in modes:
            rec = entry["modes"].get(mode)
            if rec:
                workload = f"{_shape(spec)}, {spec['out_dtype']} out, {mode}{program}"
                lines.append(_md_line(gpu, workload, spec["op"], rec, reason))
        for mode in modes:
            rec = entry.get("composite", {}).get(mode)
            if rec:
                workload = (
                    f"{_shape(spec)}, {spec['out_dtype']} weight gradient, {mode}"
                )
                lines.append(
                    _md_line(gpu, workload, "2 x fwd + dgrad + wgrad", rec, reason)
                )
    v = results["versions"]
    image = results["image"]
    lines.append("")
    lines.append(
        f"{results['device']} ({results['arch']}, {results['sm_count']} SMs), CUDA {v.get('torch_cuda')}, "
        f"driver API {v.get('cuda_driver_api')}, torch {v.get('torch')}, cuBLAS {v.get('cublas')}, "
        f"cuBLASLt {v.get('cublaslt')}, image {image.get('tag') or image.get('etc_cake_image_tag')} "
        f"({image.get('digest')}). Latency: median of {results['groups']} paired groups x {results['steps']} "
        "iterations (CUPTI GPU time of every kernel and memset of a call, cold L2), [p5, p95] of all iterations; "
        "eager = launched per iteration, graph = replayed CUDA graph. Non-Cake SOTA: the faster of the cuBLAS "
        "heuristic (torch.matmul) and the fastest correct of the cuBLASLt heuristic's top "
        f"{results.get('lt_algos')} algorithms; router: cuBLAS FP32 with CUBLAS_FP32_EMULATED_BF16X9_MATH. "
        "Speedups = baseline latency / new latency (above 1 = faster)."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Header facts
# ---------------------------------------------------------------------------


def _sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def package_identity(module):
    root = Path(module.__file__).parent
    return {
        name: _sha256(root / name)
        for name in ("cake_backend.py", "cake_jit.py", "build.json")
    } | {"path": str(root)}


def git_head(path):
    try:
        res = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return res.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def library_versions(libs):
    versions = dict(torch=torch.__version__, torch_cuda=torch.version.cuda)
    versions["cublas"] = getattr(libs.get("bf16x9"), "version", None)
    versions["cublaslt"] = getattr(libs.get("lt"), "version", None)
    try:
        from cuda.bindings import driver

        status, value = driver.cuDriverGetVersion()
        versions["cuda_driver_api"] = int(value) if int(status) == 0 else None
    except Exception as exc:  # noqa: BLE001
        versions["cuda_driver_api"] = f"unavailable: {type(exc).__name__}"
    try:
        from importlib.metadata import version

        versions["cupti_python"] = version("cupti-python")
    except Exception as exc:  # noqa: BLE001
        versions["cupti_python"] = f"unavailable: {type(exc).__name__}"
    return versions


def gpu_facts(device):
    props = torch.cuda.get_device_properties(device)
    uuid = str(getattr(props, "uuid", "") or "")
    uuid = uuid if uuid.startswith("GPU-") or not uuid else f"GPU-{uuid}"
    facts = dict(
        name=props.name,
        uuid=uuid,
        sm_count=int(props.multi_processor_count),
        capability=[int(props.major), int(props.minor)],
    )
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit,temperature.gpu",
             "--format=csv,noheader", "-i", uuid or "0"],
            capture_output=True, text=True, timeout=60,
        )  # fmt: skip
        facts["nvidia_smi"] = res.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        facts["nvidia_smi"] = f"unavailable: {type(exc).__name__}"
    return facts


def image_identity(args):
    tag_file = Path("/etc/cake-image-tag")
    return dict(
        tag=args.image_tag or os.environ.get("CAKE_IMAGE_TAG"),
        digest=args.image_digest or os.environ.get("CAKE_IMAGE_DIGEST"),
        sqsh_sha256=args.image_sqsh_sha256 or os.environ.get("CAKE_IMAGE_SQSH_SHA256"),
        etc_cake_image_tag=tag_file.read_text().strip() if tag_file.exists() else None,
        slurm_container_image=os.environ.get("SLURM_CONTAINER_IMAGE"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the numerical modes of the Cake dense projection GEMMs (the module docstring has the arms, timing and reports).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rows", nargs="*", default=None, help="row labels (default: every row)"
    )
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--ops", default=",".join(OPS))
    parser.add_argument(
        "--dtypes", default="bf16,f32", help="output dtypes of the rows to run"
    )
    parser.add_argument(
        "--T", default="", help="token counts replacing every row's own (comma list)"
    )
    parser.add_argument(
        "--only-atomic",
        action="store_true",
        help="time only the rows whose atomic arm runs this package's program",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--groups",
        type=int,
        default=5,
        help="paired groups per row (>= 5; 9 for rows within 2%%)",
    )
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--no-p1", action="store_true", help="skip the atomic_p1 arm")
    parser.add_argument(
        "--no-lt", action="store_true", help="skip the cuBLASLt sweep arm"
    )
    parser.add_argument(
        "--lt-algos", type=int, default=8, help="cuBLASLt heuristic algorithms swept"
    )
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--acc-reruns", type=int, default=3)
    parser.add_argument(
        "--memory",
        action="store_true",
        help="peak temporary and retained memory per arm",
    )
    parser.add_argument(
        "--composite",
        action="store_true",
        help="2 x forward + input gradient + weight gradient (on the weight-gradient rows)",
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help="re-time det / atomic next to a resident kernel on another stream",
    )
    parser.add_argument("--resident-sms", type=int, default=16)
    parser.add_argument("--resident-us", type=float, default=100.0)
    parser.add_argument("--host-overhead", action="store_true")
    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="require no other process on the GPU around every row (nvidia-smi)",
    )
    parser.add_argument("--exclusive-wait", type=float, default=300.0)
    parser.add_argument("--image-tag", default=None)
    parser.add_argument("--image-digest", default=None)
    parser.add_argument("--image-sqsh-sha256", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    args.modes = tuple(m for m in args.modes.split(",") if m)
    if set(args.modes) - {"eager", "graph"}:
        parser.error("--modes takes eager and / or graph")
    if args.groups < 1:
        parser.error("--groups must be >= 1")
    ops = {o for o in args.ops.split(",") if o}
    families = {f for f in args.families.split(",") if f}
    dtypes = {d for d in args.dtypes.split(",") if d}
    T_override = tuple(int(t) for t in args.T.split(",") if t) or None
    rows = row_table(T_override)
    if args.rows:
        missing = sorted(set(args.rows) - set(rows))
        if missing:
            parser.error(f"unknown rows {missing}")
        rows = {label: rows[label] for label in args.rows}
    rows = {
        label: spec
        for label, spec in rows.items()
        if spec["op"] in ops
        and spec["family"] in families
        and spec["out_dtype"] in dtypes
    }
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    args.device = device
    try:
        from cupti import cupti  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        parser.error(f"the CUPTI timer is required (cupti-python >= 13): {exc}")
    libs = {}
    try:
        libs["bf16x9"] = CublasBf16x9()
    except Exception as exc:  # noqa: BLE001 - the BF16x9 arm needs a CUDA >= 12.9 cuBLAS
        libs["bf16x9_error"] = f"{type(exc).__name__}: {exc}"
        print(f"BF16x9 arm unavailable: {libs['bf16x9_error']}", flush=True)
    try:
        libs["lt"] = CublasLt()
    except Exception as exc:  # noqa: BLE001
        libs["lt_error"] = f"{type(exc).__name__}: {exc}"
        print(f"cuBLASLt arm unavailable: {libs['lt_error']}", flush=True)
    resident = None
    if args.resident:
        try:
            resident = ResidentKernel(device, args.resident_sms, args.resident_us)
        except Exception as exc:  # noqa: BLE001 - the variant needs cuda.bindings
            print(
                f"resident variant unavailable: {type(exc).__name__}: {exc}", flush=True
            )
    gpu = gpu_facts(device)
    repo = Path(__file__).resolve().parents[1]
    results = dict(
        device=gpu["name"],
        gpu=gpu,
        sm_count=gpu["sm_count"],
        arch=det_backend.arch_for(device),
        versions=library_versions(libs),
        image=image_identity(args),
        flashinfer_head=git_head(repo),
        packages=dict(
            det=package_identity(det_backend), atomic=package_identity(cake_backend)
        ),
        bf16x9_strategy=getattr(libs.get("bf16x9"), "strategy", None),
        libs_unavailable={k: v for k, v in libs.items() if k.endswith("_error")},
        steps=args.steps,
        groups=args.groups,
        lt_algos=args.lt_algos,
        modes=list(args.modes),
        accept_rule=f"all groups atomic < det and median ratio < {ACCEPT_RATIO}, every mode",
        timer=(
            "flashinfer.testing.bench_gpu_time(enable_cupti=True, cold_l2_cache=True, use_cuda_graph=<graph>); "
            "resident: CUDA events"
        ),
        argv=sys.argv,
        rows=[],
    )
    uuid = gpu["uuid"]
    if args.exclusive:
        results["foreign_at_start"] = wait_exclusive(args, uuid)
    started = time.time()
    for label, spec in rows.items():
        results["rows"].append(measure_row(label, spec, args, libs, resident, uuid))
    results["wall_s"] = time.time() - started
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1, default=str) + "\n")
        print(f"wrote {args.json}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for entry in results["rows"]:
                for row in csv_rows(entry, args.modes):
                    writer.writerow(row)
        print(f"wrote {args.csv}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(results, args.modes))
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
