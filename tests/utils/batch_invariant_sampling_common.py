# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the batch-invariant sampling suites (``test_batch_invariant_sampling_draw.py``,
``test_batch_invariant_sampling_draw_support.py``): the device gate, the adversarial inputs, the session precompile of the
device tests' kernels on device-free workers, and the device-free compile with the PTX admission check of the IR pins."""

from __future__ import annotations

import glob
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import pytest
import torch

from flashinfer import batch_invariant as bi

#: The SM counts the dispatchers are exercised at: B200 / B300 (SM100 / SM103), Rubin (SM107) parts.
SM_COUNTS = [148, 152, 204, 224]
DS = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
#: The public entries follow the device (PDL on SM >= 9.0); the generic entries are called the same way in the suites.
PDL = True


def device_supported(archs: Iterable[Tuple[int, int]]) -> bool:
    """Whether a CUDA device of one of the class's architectures is present."""
    return torch.cuda.is_available() and bi.is_supported_device(
        torch.device("cuda"), archs
    )


def device_marker(archs: Iterable[Tuple[int, int]]):
    """The ``skipif`` marker of a class's device tests (its reason also identifies the device tests to the precompile)."""
    return pytest.mark.skipif(
        not device_supported(archs),
        reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
    )


# ------------------------------------------------------------------------------------------------ inputs
def logits(M, N, dtype, seed=0, scale=2.0**-10):
    """Rows that make reduction-order differences visible (the row class's suite): magnitudes over ~40 binades with
    random signs, near-cancelling pairs, signed zeros, subnormals, outliers, scaled into a logits-like range; finite."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    exp_span = 20.0 if dtype != torch.float16 else 6.0
    e = (torch.rand(M, N, generator=g) * 2 - 1) * exp_span
    mant = torch.rand(M, N, generator=g) + 1.0
    sign = torch.where(torch.rand(M, N, generator=g) < 0.5, -1.0, 1.0)
    x = sign * mant * torch.exp2(e)
    pairs = torch.rand(M, N // 2, generator=g) < 0.15
    idx = torch.arange(0, N - 1, 2)
    x[:, idx + 1] = torch.where(
        pairs,
        -x[:, idx] * (1 + 1e-3 * torch.randn(M, N // 2, generator=g)),
        x[:, idx + 1],
    )
    r = torch.rand(M, N, generator=g)
    outlier = 2.0**12 if dtype != torch.float16 else 2.0**4
    x = torch.where(r < 0.01, x * outlier, x)
    x = torch.where((r >= 0.01) & (r < 0.02), torch.zeros_like(x), x)
    x = torch.where((r >= 0.02) & (r < 0.03), -torch.zeros_like(x), x)
    x = torch.where((r >= 0.03) & (r < 0.035), x * 2.0**-130, x)
    x = (x.to(dtype).float() * scale).to(dtype)
    assert torch.isfinite(x).all()
    return x.cuda()


def weights(M, N, dtype, seed=0):
    """Weight rows: random, the first half of row 0 zero, row 2 one-hot at the last element."""
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(M, N, generator=g).to(dtype)
    p[0, : N // 2] = 0
    if M > 2:
        p[2] = 0
        p[2, N - 1] = 1
    return p.cuda()


# ------------------------------------------------------------------------------------------------ the precompile
def device_tests_selected(session, reason: str) -> bool:
    """Whether a collected test carries the device ``skipif`` marker with ``reason`` (otherwise the precompile has
    nothing to do: a ``-k`` run of the device-free tests on a GPU host compiles no kernel)."""
    return any(
        m.name == "skipif" and m.kwargs.get("reason") == reason
        for item in session.items
        for m in item.iter_markers()
    )


def precompile(
    specs: Sequence,
    worker_code: str,
    module: str,
    label: str,
    workers_env: str = "BI_SAMPLING_PRECOMPILE_WORKERS",
) -> None:
    """Compile ``specs`` on worker processes (``sys.executable -c worker_code <chunk>``; the worker prints ``BAD [...]``
    with the failures) that inherit this process's environment unchanged -- the same visible device, the same
    ``CUTE_DSL_ARCH`` / ``FLASHINFER_CUDA_ARCH_LIST`` -- so they compile what this process would, into its module directory,
    and a CI that validates the compiler environment of every process sees no change; asserts no failure and a filled
    module cache (``<module>_sm<cc>_cute_dsl``). ``<workers_env>`` sets the pool size (default 8; 0 disables the pool and
    the tests compile in-process); ``BI_SAMPLING_PRECOMPILE_DEVICE_FREE=1`` (a development host with many cores and no
    such validation) compiles with no device visible and the device's architecture passed explicitly instead."""
    from flashinfer.compilation_context import CompilationContext
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.cute_dsl_core import _get_compile_arch

    archs = sorted(CompilationContext().TARGET_CUDA_ARCHS)
    # the module directory's segment is the JIT's own compile target (``CUTE_DSL_ARCH`` when set, else the current
    # device's architecture with its suffix: ``sm107a``), never the lowest target of a multi-architecture list
    arch = _get_compile_arch()
    specs = list(dict.fromkeys(specs))
    random.Random(0).shuffle(
        specs
    )  # spread the heavy (cluster) kernels over the workers
    workers = int(os.environ.get(workers_env, min(8, os.cpu_count() or 1)))
    if workers <= 0 or not specs:
        print(f"{label} precompile pool off: the tests compile in-process", flush=True)
        return
    workers = max(1, min(workers, len(specs)))
    env = dict(os.environ)
    if os.environ.get("BI_SAMPLING_PRECOMPILE_DEVICE_FREE") == "1":
        # no device in the workers: the device's architecture explicitly, so the module directory
        # (FLASHINFER_CUDA_ARCH_LIST decides the workspace's segment, CUTE_DSL_ARCH the kernel target) is the one this
        # process loads from
        env.update(
            CUDA_VISIBLE_DEVICES="",
            CUTE_DSL_ARCH=os.environ.get("CUTE_DSL_ARCH") or f"sm_{arch[2:]}",
            FLASHINFER_CUDA_ARCH_LIST=" ".join(f"{a}.{b}" for a, b in archs),
        )
    chunks = [specs[i::workers] for i in range(workers)]

    def run(chunk):
        r = subprocess.run(
            [sys.executable, "-c", worker_code, repr(chunk)],
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if r.returncode != 0:
            return [(("worker",), r.stderr[-1500:])]
        tail = [ln for ln in r.stdout.splitlines() if ln.startswith("BAD ")]
        import ast

        return ast.literal_eval(tail[-1][4:]) if tail else [(("worker",), "no report")]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        bad = [b for res in ex.map(run, chunks) for b in res]
    print(
        f"{label} precompiled {len(specs)} kernels on {workers} workers in {time.time() - t0:.0f} s",
        flush=True,
    )
    assert not bad, bad[:3]
    module_dir = jit_env.FLASHINFER_JIT_DIR / f"{module}_{arch}_cute_dsl"
    n_cached = len(list(module_dir.glob("*.o"))) if module_dir.exists() else 0
    print(f"{label} {n_cached} kernels in {module_dir}", flush=True)
    assert n_cached >= len(specs) // 2, (
        "the workers did not fill this process's module cache"
    )


# ------------------------------------------------------------------------------------------------ the IR pins
#: The admitted PTX forms: the `.rn` arithmetic incl. the mixed-precision forms, the exact conversions and selects, the
#: two declared MUFU instructions, `max.f32`; integer forms are not fp-like.
PTX_FP_ALLOWED = re.compile(
    r"^(fma\.rn\.f32(\.bf16|\.f16)?|add\.rn\.f32(\.bf16|\.f16)?|sub\.rn\.f32(\.bf16|\.f16)?|mul\.rn\.f32|div\.rn\.f32"
    r"|cvt\.f32\.(bf16|f16)|cvt\.rn\.(bf16|f16)\.f32|cvt\.rn\.f32\.s32|cvt\.rn\.f32\.s64|cvt\.rn\.f32\.u32|cvt\.f32\.f16"
    r"|cvt\.rn\.f64\.u64|mul\.rn\.f64|cvt\.u32\.u64|cvt\.u64\.u32|cvt\.s64\.s32|cvt\.u64\.s32|cvt\.s32\.s64"
    r"|max\.(f32|bf16|f16)|min\.f32|abs\.f32|neg\.f32|mov\.(b32|f32)|selp\.f32|setp\.[a-z]+\.(f32|bf16|f16)"
    r"|ex2\.approx\.ftz\.f32|lg2\.approx\.ftz\.f32)$"
)
PTX_FP_LIKE = re.compile(
    r"^(fma|add|sub|mul|div|sqrt|rsqrt|rcp|ex2|lg2|cvt|max|min|abs|neg|selp|setp|fns|mad|sin|cos|tanh)\."
)


# The IR / PTX dumps are requested on the DSL's option object inside this process, not through CUTE_DSL_KEEP* environment
# variables: a CI that validates the compiler environment of every process against the one recorded at test start
# counts those as compiler switches (the DSL resolves its options from the environment once, into plain attributes of
# its singleton; the dumps are written before the cache is consulted, so they never affect what is compiled).
_IR_DUMP_PRELUDE = (
    "import os\n"
    "from cutlass.cutlass_dsl.cutlass import CuTeDSL\n"
    "_dsl = CuTeDSL()\n"
    "_dsl.envar.keep_ir = True\n"  # the raw IR before any pass (what CUTE_DSL_KEEP=ir-debug requests)
    "_dsl.envar.keep_ptx = True\n"  # the PTX (what CUTE_DSL_KEEP=ptx requests)
    "_dsl.envar.dump_dir = os.getcwd()\n"
)


def compile_device_free(tmp_path, code: str, timeout: int = 1800) -> Tuple[str, str]:
    """Run ``code`` (one kernel's compile) in a subprocess with the IR kept in ``tmp_path``; returns the MLIR and the PTX,
    after the generic checks: no ``arith`` float op or ``math`` dialect op, no fast-math, no approximate division or
    reciprocal anywhere. The subprocess inherits this process's environment plus FlashInfer's module cache off (so the DSL
    compiles and the dumps exist); the dumps are requested on the DSL's option object inside the subprocess
    (``_IR_DUMP_PRELUDE``), not through ``CUTE_DSL_KEEP*``, and with a device visible the compile targets that device as
    this process would -- a CI that validates the compiler environment of every process sees no change; without a device
    the compile is device-free with the target from ``BI_TEST_ARCH`` (default ``sm_100a``)."""
    env = dict(os.environ, FLASHINFER_CUTE_DSL_DISABLE_CACHE="1")
    if not torch.cuda.is_available():
        env.update(
            CUDA_VISIBLE_DEVICES="",
            CUTE_DSL_ARCH=os.environ.get("BI_TEST_ARCH", "sm_100a"),
        )
    r = subprocess.run(
        [sys.executable, "-c", _IR_DUMP_PRELUDE + code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert r.returncode == 0, r.stderr[-4000:]
    mlir = glob.glob(str(tmp_path / "*.mlir"))
    assert mlir, "no MLIR dumped"
    ir = Path(mlir[0]).read_text()
    assert not re.search(r"arith\.(addf|mulf|divf|subf|fmaf)\b", ir), (
        "arith float op present"
    )
    assert not re.search(r"\bmath\.", ir), "math-dialect op present"
    assert (
        "fastmath<" not in ir
        and "rsqrt" not in ir
        and "div.approx" not in ir
        and "rcp.approx" not in ir
    )
    ptxs = glob.glob(str(tmp_path / "*.ptx"))
    assert ptxs, "no PTX dumped"
    ptx = Path(ptxs[0]).read_text()
    assert not re.search(r"rcp\.|div\.approx|div\.full|sqrt\.", ptx)
    return ir, ptx


def ptx_fp_ops(ptx: str) -> set:
    """The floating-point-like PTX opcodes of a kernel (the instruction mnemonics, with their type suffixes)."""
    ops = set()
    for line in ptx.splitlines():
        tok = line.strip().split()
        if tok and tok[0].startswith("@"):
            tok = tok[1:]  # a predicated instruction: the opcode follows the predicate
        if not tok or tok[0].startswith(("//", ".", "{", "}", "bra", "ret")):
            continue
        if PTX_FP_LIKE.match(tok[0].rstrip(";")):
            ops.add(tok[0].rstrip(";"))
    return ops


#: What may stand between a PTX mnemonic and its result type: rounding and saturation modifiers, the multiply widths,
#: vector lanes, and the comparison operators of ``setp`` / ``set`` / ``slct``.
_PTX_MODIFIERS = frozenset(
    [
        "rn",
        "rz",
        "rm",
        "rp",
        "rni",
        "rzi",
        "rmi",
        "rpi",
        "sat",
        "ftz",
        "approx",
        "full",
        "lo",
        "hi",
        "wide",
    ]
    + ["x", "y", "z", "w"]
    + [
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "ls",
        "hs",
        "equ",
        "neu",
        "ltu",
        "leu",
        "gtu",
        "geu",
        "num",
        "nan",
    ]
)


def _int_result(op: str) -> bool:
    """Whether a PTX opcode's RESULT type is an integer (``cvt.rzi.s64.f32``, ``add.s32``, ``mul.wide.u32``): the first
    type suffix after the mnemonic and its modifiers. A conversion INTO a float (``cvt.rz.f32.s32``) is not exempt by
    its integer source."""
    for part in op.split(".")[1:]:
        if part in _PTX_MODIFIERS:
            continue
        return re.fullmatch(r"(b|u|s)(8|16|32|64)", part) is not None
    return False


def bad_ptx_ops(
    ptx: str, extra_allowed: Callable[[str], bool] = lambda o: False
) -> List[str]:
    """The fp-like opcodes outside the admitted forms (integer-RESULT forms, ``cvt.rzi`` and wide / lo / hi multiplies
    are admitted everywhere; ``extra_allowed`` admits a suite's own declared forms)."""
    return sorted(
        o
        for o in ptx_fp_ops(ptx)
        if not PTX_FP_ALLOWED.match(o)
        and not _int_result(o)
        and not re.match(r"^(cvt\.rzi|mul\.(wide|lo|hi))", o)
        and not extra_allowed(o)
    )
