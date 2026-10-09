# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""SM120 / SM121 implementation of ``mm_mxfp8(..., backend="cute-dsl")``.

Inputs follow the ``mm_mxfp8`` convention: ``a`` [M, K] E4M3 row-major,
``b`` [K, N] E4M3 column-major (the [N, K] weight transposed), 1D UE8M0
scales in the F8_128x4 layout. The output is BF16 or FP16.
"""

import functools
import os
import threading
from concurrent.futures import as_completed

import torch

from ....autotuner import TunableRunner
from ....utils import get_compute_capability, get_device_index
from . import policy
from .common import sf_bytes

_COMPILED: dict = {}
_LOCK = threading.Lock()


@functools.cache
def _device(index):
    props = torch.cuda.get_device_properties(index)
    return policy.Device(props.multi_processor_count, props.L2_cache_size)


def check_requirement(a, b, a_descale, b_descale, out, out_dtype, use_8x4_sf_layout):
    """Raise ``ValueError`` with a reason if the inputs are not supported."""
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("SM12x cute-dsl mm_mxfp8 supports BF16 and FP16 output only")
    if use_8x4_sf_layout or a_descale.ndim != 1 or b_descale.ndim != 1:
        raise ValueError("SM12x cute-dsl mm_mxfp8 requires 1D 128x4-swizzled scales")
    if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
        raise ValueError("SM12x cute-dsl mm_mxfp8 requires CUDA 13 or newer")
    m, k = a.shape
    n = b.shape[1]
    policy.check_shape(m, n, k)
    if a.stride() != (k, 1) or b.stride() != (1, k):
        raise ValueError(
            "SM12x cute-dsl mm_mxfp8 requires a row-major A [M, K] and a "
            "column-major B [K, N] (the transpose of a contiguous [N, K] weight)"
        )
    if out is not None and out.stride() != (n, 1):
        raise ValueError("SM12x cute-dsl mm_mxfp8 requires a contiguous output")
    tensors = (a, b, a_descale, b_descale) + (() if out is None else (out,))
    if any(t.data_ptr() % 16 for t in tensors):
        raise ValueError("SM12x cute-dsl mm_mxfp8 requires 16-byte aligned tensors")
    if not a_descale.is_contiguous() or not b_descale.is_contiguous():
        raise ValueError("SM12x cute-dsl mm_mxfp8 requires contiguous scale tensors")
    return True


class _Workspace:
    """Per-device stream-K scratch: FP32 partials and Int32 arrival counters.

    Counters are zero between launches (every kernel resets the ones it
    used). Buffers only grow; a replaced buffer stays alive because captured
    CUDA graphs may still reference it. Launches that share a device must be
    ordered on one stream.
    """

    def __init__(self, device):
        self.device = device
        self.partials = torch.empty(0, dtype=torch.float32, device=device)
        self.counters = torch.zeros(0, dtype=torch.int32, device=device)
        self._retired = []

    def get(self, n_partials, n_counters):
        if self.partials.numel() < n_partials or self.counters.numel() < n_counters:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "SM12x cute-dsl mm_mxfp8 needs a larger stream-K workspace during "
                    "CUDA graph capture; run the same shape once before capturing"
                )
            self._retired += [self.partials, self.counters]
            self.partials = torch.empty(
                max(n_partials, 2 * self.partials.numel()),
                dtype=torch.float32,
                device=self.device,
            )
            self.counters = torch.zeros(
                max(n_counters, 2 * self.counters.numel(), 1024),
                dtype=torch.int32,
                device=self.device,
            )
        return self.partials, self.counters


@functools.cache
def _workspace(index):
    return _Workspace(torch.device("cuda", index))


def _workspace_need(tactic, m, n, k, dev):
    """Scratch (partials, counters) a tactic needs for any M it can serve."""
    if tactic[0] == "skinny":
        from .skinny import Sm12xMxfp8Skinny

        _, _, nt, rt, _, _, warps, _, _, cta_wide = tactic
        grid = policy.skinny_grid(tactic, n, k, dev)
        return Sm12xMxfp8Skinny.workspace_size(
            policy.SKINNY_MAX_M, n, nt, rt, warps, grid, cta_wide
        )
    if tactic[0] == "persistent" and tactic[1] == "streamk":
        # sk_tiles * (ceil(grid / sk_tiles) + 1) <= 3 * SMs for any M.
        return 3 * dev.sms * tactic[2] * tactic[3], dev.sms
    return 0, 0


def _kernel_spec(n, k, tactic, dev, out_f16):
    """(module, kernel name, compile function, key files) of one kernel."""
    import cutlass
    import cutlass.cute as cute

    from . import common, gemv, persistent, pingpong, ptx, skinny

    family = tactic[0]
    m_sym = cute.sym_int()

    def fake(dtype, shape, align=16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(range(len(shape) - 1, -1, -1)),
            assumed_align=align,
        )

    fp8 = cutlass.Float8E4M3FN
    a = fake(fp8, (m_sym, k))
    b = fake(fp8, (n, k))
    sfa = fake(cutlass.Uint8, (cute.sym_int(),))
    sfb = fake(cutlass.Uint8, (sf_bytes(n, k),))
    out_dtype = cutlass.Float16 if out_f16 else cutlass.BFloat16
    c = fake(out_dtype, (m_sym, n), 16 if (n * 2) % 16 == 0 else 2)
    part = fake(cutlass.Float32, (cute.sym_int(),))
    cnt = fake(cutlass.Int32, (cute.sym_int(),))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if family == "gemv":
        _, _, mb, lpr, warps, _, ks = tactic
        grid = policy.gemv_grid(tactic, n, dev)
        op = gemv.Sm12xMxfp8Gemv(n, k, mb, lpr, warps, grid, ks, out_f16)
        args = (a, sfa, b, sfb, c, stream)
        source = gemv
    elif family == "skinny":
        _, _, nt, rt, ku, p, warps, _, _, cta_wide = tactic
        grid = policy.skinny_grid(tactic, n, k, dev)
        op = skinny.Sm12xMxfp8Skinny(
            n, k, nt, rt, ku, p, warps, grid, cta_wide, out_f16
        )
        args = (a, sfa, b, sfb, c, part, cnt, stream)
        source = skinny
    elif family == "persistent":
        _, _, bm, bn, wm, wn, ks, kw, sa = tactic
        sb = policy.persistent_stages(tactic)
        op = persistent.Sm12xMxfp8Persistent(
            n, k, bm, bn, wm, wn, ks, sb, kw, sa, tactic[1] == "streamk", out_f16
        )
        args = (
            a,
            b,
            sfa,
            sfb,
            c,
            part,
            cnt,
            cutlass.Int32(1),
            cutlass.Int32(0),
            stream,
        )
        grid = 0
        source = persistent
    else:
        _, _, tile_n, tile_k, epi_n, epi_stages, group, coop, early = tactic
        op = pingpong.Sm12xMxfp8PingpongGemm(
            n, k, tile_n, tile_k, (64, epi_n), epi_stages, group, coop, early, out_dtype
        )
        args = (a, b, sfa, sfb, c, cutlass.Int32(1), stream)
        grid = 0
        source = pingpong

    name = "_".join(str(int(x)) if isinstance(x, bool) else str(x) for x in tactic[1:])
    return (
        f"{policy.VERSION}_{family}",
        f"n{n}_k{k}_{name}_g{grid}_{'f16' if out_f16 else 'bf16'}",
        lambda: cute.compile(op, *args, options="--enable-tvm-ffi"),
        (__file__, policy.__file__, common.__file__, ptx.__file__, source.__file__),
    )


def _compile(n, k, tactic, dev, out_f16):
    from ....jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    module, name, compile_fn, key_files = _kernel_spec(n, k, tactic, dev, out_f16)
    return build_and_load_cute_dsl_kernel(module, name, compile_fn, key_files)


def _get_compiled(index, n, k, tactic, out_f16):
    key = (index, n, k, tactic, out_f16)
    fn = _COMPILED.get(key)
    if fn is None:
        with _LOCK:
            fn = _COMPILED.get(key)
            if fn is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "SM12x cute-dsl mm_mxfp8 compiles kernels on first use; run the "
                        "same shape once before CUDA graph capture"
                    )
                with torch.cuda.device(index):
                    fn = _compile(n, k, tactic, _device(index), out_f16)
                _COMPILED[key] = fn
    return fn


# Representative M of the autotuner buckets whose candidates are queued for
# compilation when a shape is first prepared (see _prepare).
_AHEAD_MS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
_AHEAD_DONE: set = set()
_INFLIGHT: dict = {}


def _preparation_tactics(m, n, k, dev):
    """Kernels a bucket's profiling can launch: the candidates and, for the
    cooperative ping-pong tile, its 128-row fallback for smaller M."""
    out = []
    for choice in policy.valid_tactics(m, n, k, dev):
        out.append(choice)
        if choice[0] == "pingpong" and choice[7]:
            out.append(policy.resolve(choice, 1))
    return out


def _prepare(index, n, k, tactics, out_f16):
    """Load ``tactics``, compiling the uncached ones in worker subprocesses.

    The first preparation of a shape also queues the candidates of the other
    buckets behind its own, so they compile while this bucket is profiled.
    """
    from ....jit.cute_dsl_core import (
        JitSpecCuteDsl,
        _hash_source_files,
        cute_dsl_cache_disabled,
    )
    from . import compile_pool

    shape = (index, n, k, out_f16)

    def key_of(t):
        return (index, n, k, t, out_f16)

    pending = [t for t in tactics if key_of(t) not in _COMPILED]
    if (
        pending
        and compile_pool.num_workers() > 1
        and not cute_dsl_cache_disabled()
        and not torch.cuda.is_current_stream_capturing()
    ):
        dev = _device(index)
        wanted = list(pending)
        if shape not in _AHEAD_DONE:
            _AHEAD_DONE.add(shape)
            for m in _AHEAD_MS:
                wanted += _preparation_tactics(m, n, k, dev)
        jobs, queued = [], set()
        for t in wanted:
            key = key_of(t)
            if key in _COMPILED or key in _INFLIGHT or key in queued:
                continue
            module, name, compile_fn, key_files = _kernel_spec(n, k, t, dev, out_f16)
            spec = JitSpecCuteDsl(
                module, name, compile_fn, _hash_source_files(key_files)
            )
            if not spec.is_compiled:
                jobs.append(t)
                queued.add(key)
        if jobs:
            major, minor = get_compute_capability(torch.device("cuda", index))
            arch = os.environ.get("CUTE_DSL_ARCH") or f"sm_{major}{minor}a"
            futures = compile_pool.submit(
                [[n, k, list(t), dev.sms, dev.l2_bytes, out_f16] for t in jobs],
                arch,
                start=len(jobs) > 1,
            )
            if futures is not None:
                for t, fut in zip(jobs, futures, strict=True):
                    _INFLIGHT[key_of(t)] = fut
        waits = {_INFLIGHT[key_of(t)]: t for t in pending if key_of(t) in _INFLIGHT}
        for fut in as_completed(waits):
            err = fut.result()
            if err is not None:
                compile_pool.logger.debug(f"SM12x mxfp8 compile worker: {err}")
            _get_compiled(index, n, k, waits[fut], out_f16)
    for t in tactics:
        _INFLIGHT.pop(key_of(t), None)
        _get_compiled(index, n, k, t, out_f16)


def launch(tactic, a, b, a_descale, b_descale, out):
    """Run one tactic: out[M, N] = a[M, K] @ b[K, N] with block scales."""
    m, k = a.shape
    n = b.shape[1]
    if m == 0:
        return out
    index = get_device_index(a.device)
    dev = _device(index)
    fn = _get_compiled(index, n, k, tactic, out.dtype == torch.float16)
    w = b.T
    family = tactic[0]
    if family == "gemv":
        fn(a, a_descale, w, b_descale, out)
    elif family == "skinny":
        part, cnt = _workspace(index).get(*_workspace_need(tactic, m, n, k, dev))
        fn(a, a_descale, w, b_descale, out, part, cnt)
    elif family == "persistent":
        part, cnt = _workspace(index).get(*_workspace_need(tactic, m, n, k, dev))
        grid, sk_tiles = policy.persistent_schedule(tactic, m, n, k, dev)
        fn(a, w, a_descale, b_descale, out, part, cnt, grid, sk_tiles)
    else:
        cta_m = 256 if tactic[7] else 128
        tiles = -(-m // cta_m) * -(-n // tactic[2])
        fn(a, w, a_descale, b_descale, out, min(dev.sms, tiles))
    return out


class Sm12xMxfp8GemmRunner(TunableRunner):
    """``TunableRunner`` over every SM12x MXFP8 kernel family (see policy.py)."""

    def get_cache_key_extras(self, inputs):
        a, _, _, _, _, out, _ = inputs
        return (policy.VERSION, get_compute_capability(a.device), str(out.dtype))

    def get_valid_tactics(self, inputs, profile):
        a, b = inputs[:2]
        dev = _device(get_device_index(a.device))
        return policy.valid_tactics(a.shape[0], b.shape[1], a.shape[1], dev)

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, a_descale, b_descale, _, out, _ = inputs
        m, k = a.shape
        n = b.shape[1]
        index = get_device_index(a.device)
        dev = _device(index)
        if do_preparation:
            tactics = _preparation_tactics(m, n, k, dev)
            for choice in tactics:
                _workspace(index).get(*_workspace_need(choice, m, n, k, dev))
            _prepare(index, n, k, tactics, out.dtype == torch.float16)
            return out
        if tactic is None or tactic == -1 or not policy.supports_m(tactic, m):
            tactic = policy.default_tactic(m, n, k, dev)
        return launch(policy.resolve(tactic, m), a, b, a_descale, b_descale, out)


@functools.cache
def get_runner():
    return Sm12xMxfp8GemmRunner()
