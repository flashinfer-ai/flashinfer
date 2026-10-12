# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Cached CuTe DSL launch and bounded native W4A16 tuning."""

import os
from concurrent.futures import as_completed
from functools import cache
from typing import Any

import torch

from ....autotuner import AutoTuner, TunableRunner, TuningConfig
from ....utils import get_compute_capability, get_device_index
from . import grouped, prefetch

_COMPILED: dict[tuple, Any] = {}
_TUNING_CONFIG = TuningConfig(use_cuda_graph=True, use_cold_l2_cache=True)


def _kernel_spec(m, n, k, dtype, has_alpha, enable_pdl, tactic):
    """(module, kernel name, compile function, key files) of one kernel."""
    import cutlass
    import cutlass.cute as cute

    from ....cute_dsl import fp4_common
    from . import kernel, staged_kernel, tiled_staged_kernel

    kind, extent, warps, splits = tactic[:4]
    if kind == "n64":
        from .n64_compile import _kernel_spec as n64_spec

        if dtype != torch.bfloat16 or not has_alpha:
            raise ValueError("The 64-column tactic needs BF16 output and alpha")
        return n64_spec(m, n, k, enable_pdl, tactic[1:])
    staged = kind in ("staged", "tiled")
    out_type = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    a = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32 if staged else cutlass.BFloat16,
        (m, k // 2 if staged else k),
        stride_order=(1, 0),
        assumed_align=16 if staged else 2,
    )
    b = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (n, k // 8),
        stride_order=(1, 0),
        assumed_align=16 if staged else 4,
    )
    sf_size = ((n + 127) // 128) * ((k // 16 + 3) // 4) * 512
    sf = cute.runtime.make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (sf_size,), assumed_align=16 if staged else 1
    )
    alpha = (
        cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        if has_alpha
        else None
    )
    out = cute.runtime.make_fake_compact_tensor(
        out_type, (m, n), stride_order=(1, 0), assumed_align=2
    )
    if kind == "tiled":
        op = tiled_staged_kernel.NativeBf16Fp4TiledStagedKernel(
            splits, warps, enable_pdl, extent, *tactic[4:]
        )
    elif staged:
        op = staged_kernel.NativeBf16Fp4StagedKernel(
            splits, warps, enable_pdl, extent, tactic[4]
        )
    elif kind == "mma":
        op = kernel.NativeBf16Fp4MmaKernel(splits, warps, enable_pdl, extent)
    else:
        op = kernel.NativeBf16Fp4Kernel(extent, warps, enable_pdl, splits)
    partial = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (splits, m, n),
            stride_order=(2, 1, 0),
            assumed_align=4,
        )
        if splits > 1
        else None
    )
    operands = (a, b, sf, alpha, out, partial)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    dtype_key = "bf16" if dtype == torch.bfloat16 else "fp16"
    name = f"m{m}_n{n}_k{k}_{dtype_key}_a{int(has_alpha)}_p{int(enable_pdl)}_{kind}{extent}_w{warps}_s{splits}"
    if staged:
        name += f"_st{tactic[4]}"
    if kind == "tiled":
        name += f"_tm{tactic[5]}_rm{int(tactic[6])}"
    return (
        "native_bf16_fp4_sm12x",
        name,
        lambda: cute.compile(op, *operands, stream, options="--enable-tvm-ffi"),
        (
            __file__,
            kernel.__file__,
            staged_kernel.__file__,
            tiled_staged_kernel.__file__,
            fp4_common.__file__,
        ),
    )


def _compile(m, n, k, dtype, has_alpha, enable_pdl, tactic):
    from ....jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    module, name, compile_fn, key_files = _kernel_spec(
        m, n, k, dtype, has_alpha, enable_pdl, tactic
    )
    return build_and_load_cute_dsl_kernel(module, name, compile_fn, key_files)


# Powers of two up to the small-M tactic limit, the M values a serving engine
# usually tunes. The native kernels specialize on M, so the first preparation
# of a shape queues the candidates of all of them behind its own.
_AHEAD_MS = (1, 2, 4, 8, 16)
_AHEAD_DONE: set = set()
_INFLIGHT: dict = {}


def _prepare(inputs, tactics, ahead):
    """Compile the uncached ``tactics`` of ``inputs`` in worker subprocesses.

    ``ahead`` lists ``(m, tactics)`` of other buckets queued behind them on
    the first call for a shape, so they compile while this bucket profiles.
    The kernels are loaded later by ``forward``, from the disk cache.
    """
    from ....jit.cute_dsl_core import (
        JitSpecCuteDsl,
        _hash_source_files,
        cute_dsl_cache_disabled,
    )
    from . import compile_pool

    a, b, _, alpha, out, enable_pdl = inputs
    m, k = a.shape
    n = b.shape[0]
    index = get_device_index(a.device)
    if (
        compile_pool.num_workers() <= 1
        or cute_dsl_cache_disabled()
        or torch.cuda.is_current_stream_capturing()
    ):
        return
    common = (n, k, out.dtype, alpha is not None, enable_pdl)

    def key_of(mm, t):
        return (index, mm, n, k, out.dtype, alpha is None, enable_pdl, t)

    wanted = [(m, t) for t in tactics]
    shape = (index, *common)
    if shape not in _AHEAD_DONE:
        _AHEAD_DONE.add(shape)
        wanted += [(mm, t) for mm, ts in ahead for t in ts]
    jobs, keys = [], []
    for mm, t in wanted:
        key = key_of(mm, t)
        if key in _COMPILED or key in _INFLIGHT or key in keys:
            continue
        module, name, compile_fn, key_files = _kernel_spec(mm, *common, t)
        spec = JitSpecCuteDsl(module, name, compile_fn, _hash_source_files(key_files))
        if not spec.is_compiled:
            jobs.append([mm, n, k, out.dtype == torch.float16, *common[3:], list(t)])
            keys.append(key)
    if len(jobs) > 1:
        major, minor = get_compute_capability(a.device)
        arch = os.environ.get("CUTE_DSL_ARCH") or f"sm_{major}{minor}a"
        futures = compile_pool.submit(jobs, arch)
        if futures is not None:
            _INFLIGHT.update(zip(keys, futures, strict=True))
    waits = [_INFLIGHT[key_of(m, t)] for t in tactics if key_of(m, t) in _INFLIGHT]
    errors = [err for fut in as_completed(waits) if (err := fut.result()) is not None]
    if errors:
        compile_pool.logger.warning(
            f"Native W4A16: {len(errors)} worker compiles failed, compiling "
            f"them in process. First error: {errors[0]}"
        )
    for t in tactics:
        _INFLIGHT.pop(key_of(m, t), None)


def _can_stage(inputs):
    a, b, sf = inputs[:3]
    return a.shape[1] % 64 == 0 and all(t.data_ptr() % 16 == 0 for t in (a, b, sf))


def _tactics(m, n, k, can_stage, bf16_alpha, for_tuning):
    """Prune fallback tactics for tuning without rejecting safe execution."""
    if m > 16:
        tactics = [("mma", 4 if n >= 512 else 1, w, 1) for w in (4, 8)]
        if can_stage:
            tactics += [
                ("tiled", tk, w, s, stages, tm, raster_m)
                for tm, tk, w, s, stages, raster_m in (
                    (64, 64, 8, 1, 3, False),
                    (64, 128, 8, 2, 2, False),
                    (128, 64, 4, 1, 2, False),
                    (128, 128, 8, 1, 2, False),
                    (64, 64, 4, 1, 3, True),
                    (128, 128, 8, 1, 2, True),
                )
                if k % tk == 0 and s <= k // tk
            ]
        return tactics
    rows = sorted({1, min(m, 2), min(m, 4)})
    simd_splits = [s for s in (1, 2, 4, 8) if s <= (k + 255) // 256] if m == 1 else [1]
    tactics = [("simd", r, w, s) for r in rows for w in (4, 8) for s in simd_splits]
    if can_stage:
        if m > 1 and for_tuning:
            tactics = []
        tactics += [
            ("staged", tk, w, s, stages)
            for tk in (128 if k % 128 == 0 else 64,)
            for w in (4, 8)
            for s in (1, 2, 4, 8)
            if s <= k // tk
            for stages in (2, 3)
        ]
        if m <= 4 and bf16_alpha:
            tactics += [
                ("n64", tk, w, s, stages)
                for tk in (128, 256)
                if k % tk == 0
                for w in (4, 8)
                for s in (1, 2, 4, 8)
                if s <= k // tk
                for stages in (2, 3)
                if stages
                * (
                    (m + 1) * (tk // 2 + 4) * 4
                    + 64 * (tk // 8 + 4) * 4
                    + (tk // 64) * 512
                )
                <= 49152
            ]
        if for_tuning:
            return tactics
    mma_splits = (1, 2, 4, 8, 16, 32, 64) if n < 8192 and k >= 4096 else (1, 2, 4, 8)
    tactics += [
        ("mma", 4 if n >= 512 else 1, w, s)
        for s in mma_splits
        if s <= (k + 63) // 64
        for w in (4, 8)
    ]
    return tactics


class NativeBf16Fp4Runner(TunableRunner):
    def get_cache_key_extras(self, inputs):
        a, _, _, alpha, out, enable_pdl = inputs
        return (
            get_compute_capability(a.device),
            out.dtype,
            alpha is None,
            enable_pdl,
            _can_stage(inputs),
            prefetch.is_supported(inputs),
            grouped.is_supported(inputs),
        )

    def get_valid_tactics(self, inputs, profile):
        tactics = self._get_tactics(inputs, for_tuning=True)
        if prefetch.is_supported(inputs):
            tactics.append(("prefetch",))
        if grouped.is_supported(inputs):
            tactics.append(("grouped",))
        return tactics

    def _get_tactics(self, inputs, *, for_tuning):
        a, b, _, alpha, out, _ = inputs
        m, k = a.shape
        bf16_alpha = out.dtype == torch.bfloat16 and alpha is not None
        return _tactics(m, b.shape[0], k, _can_stage(inputs), bf16_alpha, for_tuning)

    def validate_tactic(self, inputs, tactic):
        # Cold-L2 profiling clones can be more aligned than the caller's views.
        if tactic == ("grouped",):
            return grouped.is_supported(inputs)
        if tactic == ("prefetch",):
            return prefetch.is_supported(inputs)
        return tactic == -1 or tactic in self._get_tactics(inputs, for_tuning=False)

    def precompile_tactics(self, inputs, tactics, profile, **kwargs):
        a, b, _, alpha, out, _ = inputs
        m, k = a.shape
        if m <= 16:
            n = b.shape[0]
            can_stage = _can_stage(inputs)
            bf16_alpha = out.dtype == torch.bfloat16 and alpha is not None
            _prepare(
                inputs,
                [t for t in tactics if t != -1 and len(t) > 1],
                [
                    (mm, _tactics(mm, n, k, can_stage, bf16_alpha, True))
                    for mm in _AHEAD_MS
                    if mm != m
                ],
            )
        return False

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, sf, alpha, out, enable_pdl = inputs
        m, k = a.shape
        n = b.shape[0]
        if not self.validate_tactic(inputs, tactic):
            raise ValueError("Invalid native W4A16 tactic")
        if tactic == ("grouped",):
            return grouped.run(a, b, sf, alpha, out, do_preparation=do_preparation)
        if tactic == ("prefetch",):
            return prefetch.run(a, b, sf, alpha, out, do_preparation=do_preparation)
        if tactic == -1:
            if m > 16:
                tactic = (
                    ("tiled", 64, 4, 1, 3, 64, True)
                    if _can_stage(inputs)
                    else ("mma", 4 if n >= 512 else 1, 4, 1)
                )
            elif _can_stage(inputs):
                tk = 128 if k % 128 == 0 else 64
                splits = min(4 if n < 8192 else 1, k // tk)
                tactic = ("staged", tk, 8 if n < 8192 else 4, splits, 3)
            elif m == 1:
                tactic = ("simd", 1, 4, 2 if n < 8192 and k >= 512 else 1)
            else:
                splits = max(s for s in (1, 2, 4, 8) if s <= (k + 63) // 64)
                if n < 8192 and k >= 4096:
                    splits = 32
                tactic = ("mma", 4 if n >= 512 else 1, 4, splits)
        key = (
            get_device_index(a.device),
            m,
            n,
            k,
            out.dtype,
            alpha is None,
            enable_pdl,
            tactic,
        )
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm cute-dsl-native for this shape before graph capture"
                )
            with torch.cuda.device(a.device):
                compiled = _compile(
                    m, n, k, out.dtype, alpha is not None, enable_pdl, tactic
                )
            _COMPILED[key] = compiled
        operands = [
            a.view(torch.int32) if tactic[0] in ("staged", "n64", "tiled") else a,
            b.view(torch.int32),
            sf.view(torch.float8_e4m3fn).view(-1),
            alpha,
            out,
        ]
        splits = tactic[3]
        partial = (
            torch.empty((splits, m, n), device=a.device, dtype=torch.float32)
            if splits > 1
            else None
        )
        operands.append(partial)
        compiled(*operands)
        return out


@cache
def get_runner():
    return NativeBf16Fp4Runner()


def run(a, b, sf, alpha, out_dtype, out, enable_pdl):
    if out is None:
        out = torch.empty((a.shape[0], b.shape[0]), dtype=out_dtype, device=a.device)
    inputs = [a, b, sf, alpha, out, enable_pdl]
    runner, tactic = AutoTuner.get().choose_one(
        "native_bf16_fp4_sm12x", [get_runner()], _TUNING_CONFIG, inputs
    )
    return runner(inputs, tactic=tactic)
