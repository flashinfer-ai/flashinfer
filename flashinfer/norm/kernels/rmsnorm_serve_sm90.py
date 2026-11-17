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

SM90 serve-shape RMSNorm fast path (H100/H200 decode-serve envelope)
====================================================================

An alternative 2D bf16 RMSNorm implementation for the launch-bound
decode-serve envelope on SM90, served through the ``"specialized"`` backend
of :func:`flashinfer.norm.rmsnorm` (#5667 pattern):

- ``backend="specialized"`` selects it explicitly; a call it cannot serve
  falls back to the stock rmsnorm dispatch (this op's composable path) —
  never to another fused backend — with the reason recorded in the dispatch
  stats.
- ``backend="auto"`` (the default) routes to ``"specialized"`` only on the
  dispatch allowlist (``flashinfer/rmsnorm_serve_sm90_dispatch.json``).
- ``backend="stock"`` pins the stock dispatch; an explicitly selected
  backend is never silently swapped, and unknown names raise.

Behind ``"specialized"`` sit TWO agent-generated kernel variants (one
backend string, per #5667, which also shipped multiple sources):

- ``cutedsl`` (default variant): a CuTe DSL kernel whose per-H plan
  (``_plan``) picks vector width, thread count and values-per-thread from H
  itself.  It is generic over 16-byte-vectorizable H; dispatch admits the
  measured band (H a multiple of 8 in [512, 6144]).
- ``cuda`` (variant opt-in via ``FLASHINFER_SM90_SERVE_RMSNORM=cuda``): a
  CUDA C++ specialist (csrc/norm_serve_sm90.cu, JIT module
  ``norm_serve_sm90``) with shape-specialized kernels for H in exactly
  {512, 2048, 6144}; dispatch admits only that exact allowlist (its generic
  fallback is correct but slower than the stock path off-list).

Kill switch: ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` (checked per call;
shared by FlashInfer's specialized kernels) forces the stock path.
``FLASHINFER_SM90_SERVE_RMSNORM=0`` remains a deprecated kill alias for one
release.  Any call outside the guard (shape, dtype, layout, architecture,
token count, CUDA-graph or autotune constraints below) also falls back to
the stock path.

CUDA-graph rules: the compiled kernels take their stream from the caller's
current stream on every call (TVM-FFI env stream / per-call stream query), so
capture goes to the capture stream; no variant is JIT-compiled under capture
(an uncompiled (H, dtype) during capture falls back to the stock path, which
has the same no-JIT-under-capture discipline once warmed); autotune tuning
mode bails to the stock path so tuning measures stock tactics only.
"""

import os

import torch

# Envelope derived from the kernels' own guard relations and the measured
# win envelope (12 timed serve rows + the off-list H=4096 generality probe;
# correctness additionally stressed at T=8192).
_SERVE_H_ALLOWLIST = (512, 2048, 6144)
_GENERIC_H_MIN = 512
_GENERIC_H_MAX = 6144
_MAX_TOKENS = 8192

_ENV_VAR = "FLASHINFER_SM90_SERVE_RMSNORM"
# Kill switch shared by FlashInfer's specialized kernels (read per call).
_SPECIALIZED_KERNEL_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
# Backends of flashinfer.norm.rmsnorm, as plain strings (#5667): "specialized"
# is the agent-generated fast path (both variants), "stock" the existing
# dispatch. An explicitly selected backend is never silently swapped.
RMSNORM_BACKENDS = ("auto", "specialized", "stock")
_DISPATCH_FILE = "rmsnorm_serve_sm90_dispatch.json"

# Per-backend dispatch census and fallback reasons (#5667-style).
_stats: dict = {
    "calls": 0,
    "backend_dispatches": {"specialized": 0, "stock": 0},
    "variant_dispatches": {"cutedsl": 0, "cuda": 0},
    "fallback_reasons": {},
}


def _fallback(reason):
    _stats["fallback_reasons"][reason] = _stats["fallback_reasons"].get(reason, 0) + 1


def _rmsnorm_serve_sm90_stats() -> dict:
    return {
        "calls": _stats["calls"],
        "backend_dispatches": dict(_stats["backend_dispatches"]),
        "variant_dispatches": dict(_stats["variant_dispatches"]),
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        "backends": RMSNORM_BACKENDS,
        "dispatch_table_loaded": _dispatch_table() is not None,
    }


def _reset_rmsnorm_serve_sm90_stats() -> None:
    _stats["calls"] = 0
    for d in (_stats["backend_dispatches"], _stats["variant_dispatches"]):
        for k in d:
            d[k] = 0
    _stats["fallback_reasons"].clear()


def _dispatch_table():
    """Load the dispatch surface (package data) once; ``None`` means
    ``backend="auto"`` never routes to "specialized"."""
    global _DISPATCH_TABLE
    try:
        return _DISPATCH_TABLE
    except NameError:
        pass
    try:
        import json as _json
        from importlib import resources

        payload = _json.loads(
            resources.files("flashinfer").joinpath(_DISPATCH_FILE).read_text()
        )
        _DISPATCH_TABLE = payload if isinstance(payload, dict) else None
    except Exception:
        _DISPATCH_TABLE = None
    return _DISPATCH_TABLE


try:
    import math

    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    from cutlass import Float32, Int32, const_expr
    import cuda.bindings.driver as cuda  # noqa: F401 (CUstream type for cute)

    _HAS_CUTE_DSL = True
except ImportError:
    _HAS_CUTE_DSL = False


if _HAS_CUTE_DSL:
    _TORCH2CUTE = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
    }

    MAX_THREADS = 384
    MIN_VPT = 2

    def _plan(h, max_threads=MAX_THREADS):
        vec = math.gcd(h, 8)
        nv = h // vec
        threads = min(max_threads, max(32, ((nv + 31) // 32) * 32))
        for cand in (384, 256, 192, 128, 96, 64, 32):
            if cand <= max_threads and nv % cand == 0 and nv // cand >= MIN_VPT:
                threads = cand
                break
        vpt = (nv + threads - 1) // threads
        exact = vpt * threads == nv
        return vec, nv, threads, vpt, exact

    @cute.kernel
    def _rmsnorm_kernel(
        xv: cute.Tensor,
        wv: cute.Tensor,
        ov: cute.Tensor,
        eps: Float32,
        VEC: cutlass.Constexpr,
        THREADS: cutlass.Constexpr,
        VPT: cutlass.Constexpr,
        NV: cutlass.Constexpr,
        NWARP: cutlass.Constexpr,
        INV_H: cutlass.Constexpr,
        EXACT: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        idxs = []
        frx = []
        for j in cutlass.range_constexpr(VPT):
            idx = Int32(j * THREADS) + tidx
            sidx = idx if const_expr(EXACT) else (idx if idx < NV else Int32(0))
            idxs.append((idx, sidx))
            src = xv[(bidx, sidx, None)]
            f = cute.make_fragment_like(src)
            cute.autovec_copy(src, f)
            frx.append(f)

        frw = []
        for j in cutlass.range_constexpr(VPT):
            wsrc = wv[(idxs[j][1], None)]
            wf = cute.make_fragment_like(wsrc)
            cute.autovec_copy(wsrc, wf)
            frw.append(wf)

        acc = Float32(0.0)
        frxf = []
        for j in cutlass.range_constexpr(VPT):
            xf = frx[j].load().to(Float32)
            frxf.append(xf)
            sum0 = Float32(0.0)
            sum1 = Float32(0.0)
            for k in cutlass.range_constexpr(VEC):
                if const_expr(k % 2 == 0):
                    sum0 = cute.fma(xf[k], xf[k], sum0)
                else:
                    sum1 = cute.fma(xf[k], xf[k], sum1)
            part = sum0 + sum1
            if const_expr(EXACT):
                acc = acc + part
            else:
                acc = acc + (part if idxs[j][0] < NV else Float32(0.0))

        total = cute.arch.warp_reduction_sum(acc)
        if const_expr(NWARP > 1):
            smem = cutlass_utils.SmemAllocator()
            lane = cute.arch.lane_idx()
            warp = cute.arch.warp_idx()
            if const_expr(NWARP <= 12):
                red = smem.allocate_tensor(
                    Float32, cute.make_layout(NWARP), byte_alignment=4
                )
                if lane == 0:
                    red[warp] = total
                cute.arch.barrier()
                if const_expr(NWARP == 12):
                    s01 = red[0] + red[1]
                    s23 = red[2] + red[3]
                    s45 = red[4] + red[5]
                    s67 = red[6] + red[7]
                    s89 = red[8] + red[9]
                    sab = red[10] + red[11]
                    total = ((s01 + s23) + (s45 + s67)) + (s89 + sab)
                else:
                    total = red[0]
                    for k in cutlass.range_constexpr(1, NWARP):
                        total = total + red[k]
            else:
                red = smem.allocate_tensor(
                    Float32, cute.make_layout(32), byte_alignment=4
                )
                if lane == 0:
                    red[warp] = total
                if const_expr(NWARP < 32):
                    if tidx < Int32(32 - NWARP):
                        red[Int32(NWARP) + tidx] = Float32(0.0)
                cute.arch.barrier()
                total = cute.arch.warp_reduction_sum(red[lane])

        rstd = cute.math.rsqrt(total * Float32(INV_H) + eps, fastmath=True)

        for j in cutlass.range_constexpr(VPT):
            xf = frxf[j]
            wf = frw[j].load().to(Float32)
            fo = cute.make_fragment_like(frx[j])
            fo.store((xf * rstd * wf).to(fo.element_type))
            if const_expr(EXACT):
                cute.autovec_copy(fo, ov[(bidx, idxs[j][1], None)])
            else:
                if idxs[j][0] < NV:
                    cute.autovec_copy(fo, ov[(bidx, idxs[j][1], None)])

    @cute.jit
    def _rmsnorm_host(
        mX: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        eps: Float32,
        stream: cuda.CUstream,
        H: cutlass.Constexpr,
        VEC: cutlass.Constexpr,
        THREADS: cutlass.Constexpr,
        VPT: cutlass.Constexpr,
        NV: cutlass.Constexpr,
        EXACT: cutlass.Constexpr,
    ):
        t = mX.shape[0]
        xv = cute.make_tensor(
            mX.iterator, cute.make_layout((t, NV, VEC), stride=(H, VEC, 1))
        )
        ov = cute.make_tensor(
            mO.iterator, cute.make_layout((t, NV, VEC), stride=(H, VEC, 1))
        )
        wv = cute.make_tensor(mW.iterator, cute.make_layout((NV, VEC), stride=(VEC, 1)))
        nwarp = THREADS // 32
        _rmsnorm_kernel(
            xv, wv, ov, eps, VEC, THREADS, VPT, NV, nwarp, 1.0 / H, EXACT
        ).launch(
            grid=[t, 1, 1],
            block=[THREADS, 1, 1],
            smem=128 if nwarp > 1 else 0,
            stream=stream,
        )

    _COMPILED: dict = {}

    def _build(h, torch_dtype):
        dt = _TORCH2CUTE[torch_dtype]
        vec, nv, threads, vpt, exact = _plan(h)
        batch = cute.sym_int()
        align = min(16, (vec * dt.width) // 8)
        fx = cute.runtime.make_fake_tensor(
            dt, (batch, h), stride=(h, 1), assumed_align=align
        )
        fo = cute.runtime.make_fake_tensor(
            dt, (batch, h), stride=(h, 1), assumed_align=align
        )
        fw = cute.runtime.make_fake_tensor(dt, (h,), stride=(1,), assumed_align=align)
        return cute.compile(
            _rmsnorm_host,
            fx,
            fw,
            fo,
            Float32(0.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            h,
            vec,
            threads,
            vpt,
            nv,
            exact,
            options="--enable-tvm-ffi",
        )


# ---------------------------------------------------------------------------
# CUDA C++ specialist backend (lazy JIT module)
# ---------------------------------------------------------------------------

_cuda_module = None


def _get_cuda_module():
    global _cuda_module
    if _cuda_module is None:
        from ...jit.norm import gen_norm_serve_sm90_module

        _cuda_module = gen_norm_serve_sm90_module().build_and_load()
    return _cuda_module


# ---------------------------------------------------------------------------
# Dispatch guard
# ---------------------------------------------------------------------------


def _disabled() -> bool:
    if os.environ.get(_SPECIALIZED_KERNEL_DISABLE_ENV, "").strip() not in (
        "",
        "0",
    ):
        return True
    value = os.environ.get(_ENV_VAR, "").strip().lower()
    if value in ("0", "off", "false", "disable", "disabled"):
        # deprecated kill alias (one release); the variant selector role of
        # this env ("cuda") is unaffected
        return True
    return False


def _variant():
    """Kernel variant behind the "specialized" backend: ``"cutedsl"``
    (default) or ``"cuda"`` (opt-in via FLASHINFER_SM90_SERVE_RMSNORM=cuda).
    Read per call."""
    if os.environ.get(_ENV_VAR, "").strip().lower() == "cuda":
        return "cuda"
    return "cutedsl"


def _h_in_envelope(h: int, backend: str) -> bool:
    if h in _SERVE_H_ALLOWLIST:
        return True
    if backend == "cutedsl":
        # Demonstrated-generic band: fully vectorized plans (H % 8 == 0)
        # inside the measured H range (off-list probe H=4096).
        return h % 8 == 0 and _GENERIC_H_MIN <= h <= _GENERIC_H_MAX
    # The CUDA specialist wins only on its exact shape-specialized H set.
    return False


def _autotune_active() -> bool:
    try:
        from ...autotuner import AutoTuner

        return bool(AutoTuner.get().is_tuning_mode)
    except Exception:
        return False


def try_rmsnorm_serve_sm90(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    requested: str = "auto",
) -> bool:
    """Run the SM90 serve-shape RMSNorm when the call is inside the measured
    envelope.  Returns ``True`` when the output has been produced; ``False``
    means the caller must run the stock path (this op's composable path:
    guard miss, kill switch, backend="stock", off the auto allowlist,
    missing runtime, uncompiled variant under CUDA-graph capture, or
    autotune tuning mode).  ``requested`` is the public backend kwarg:
    "stock" never reaches these kernels; "specialized" is explicit (guards
    still apply, with the miss reason recorded); "auto" additionally
    requires the dispatch allowlist to be loaded."""
    _stats["calls"] += 1
    if requested == "stock":
        _stats["backend_dispatches"]["stock"] += 1
        return False
    if _disabled():
        _fallback(f"{requested}: kill switch")
        return False
    if requested == "auto" and _dispatch_table() is None:
        _fallback("auto: dispatch allowlist unavailable")
        return False
    backend = _variant()
    if input.dim() != 2:
        _fallback(f"{requested}: input dim {input.dim()}")
        return False
    tokens, hidden = input.shape
    if (
        input.dtype is not torch.bfloat16
        or weight.dtype is not torch.bfloat16
        or out.dtype is not torch.bfloat16
    ):
        _fallback(f"{requested}: non-bf16 dtypes")
        return False
    if not 0 < tokens <= _MAX_TOKENS:
        _fallback(f"{requested}: tokens {tokens} outside envelope")
        return False
    if not _h_in_envelope(hidden, backend):
        _fallback(f"{requested}: H {hidden} off the allowlist")
        return False
    if input.device.type != "cuda":
        _fallback(f"{requested}: non-CUDA device")
        return False
    if weight.shape != (hidden,) or out.shape != input.shape:
        return False
    if weight.device != input.device or out.device != input.device:
        return False
    if not (input.is_contiguous() and out.is_contiguous() and weight.is_contiguous()):
        return False
    if (input.data_ptr() | out.data_ptr() | weight.data_ptr()) % 16 != 0:
        return False
    if torch.cuda.get_device_capability(input.device) != (9, 0):
        _fallback(f"{requested}: not SM90")
        return False
    if _autotune_active():
        _fallback(f"{requested}: autotuner measurement window")
        return False

    if backend == "cuda":
        if _cuda_module is None and torch.cuda.is_current_stream_capturing():
            # Never JIT-build the module under CUDA-graph capture.
            _fallback(f"{requested}: cuda variant uncompiled under capture")
            return False
        try:
            module = _get_cuda_module()
        except Exception:
            _fallback(f"{requested}: cuda variant JIT build failed")
            return False
        module.rmsnorm_serve_sm90(out, input, weight, float(eps))
        _stats["backend_dispatches"]["specialized"] += 1
        _stats["variant_dispatches"]["cuda"] += 1
        return True

    if not _HAS_CUTE_DSL:
        _fallback(f"{requested}: CuTe DSL runtime unavailable")
        return False
    key = (hidden, input.dtype)
    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            # Never JIT-compile a new (H, dtype) variant under capture.
            _fallback(f"{requested}: cutedsl variant uncompiled under capture")
            return False
        try:
            fn = _build(hidden, input.dtype)
        except Exception:
            _fallback(f"{requested}: cutedsl variant JIT build failed")
            return False
        _COMPILED[key] = fn
    # The compiled kernel takes the TVM-FFI env stream, i.e. the caller's
    # current torch stream re-queried on every call (CUDA-graph safe).
    fn(input, weight, out, float(eps))
    _stats["backend_dispatches"]["specialized"] += 1
    _stats["variant_dispatches"]["cutedsl"] += 1
    return True


__all__ = ["RMSNORM_BACKENDS", "try_rmsnorm_serve_sm90"]
