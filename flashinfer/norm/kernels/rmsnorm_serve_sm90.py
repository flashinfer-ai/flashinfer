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
decode-serve envelope on SM90: small-to-mid token counts at serve hidden
sizes.  Two backends:

- ``cutedsl`` (default): a CuTe DSL kernel whose per-H plan (``_plan``) picks
  vector width, thread count and values-per-thread from H itself.  It is
  generic over 16-byte-vectorizable H; dispatch admits the measured band
  (H a multiple of 8 in [512, 6144]).
- ``cuda`` (opt-in via ``FLASHINFER_SM90_SERVE_RMSNORM=cuda``): a CUDA C++
  specialist (csrc/norm_serve_sm90.cu, JIT module ``norm_serve_sm90``) with
  shape-specialized kernels for H in exactly {512, 2048, 6144}; dispatch
  admits only that exact allowlist (its generic fallback is correct but
  slower than the stock path off-list).

Kill switch: ``FLASHINFER_SM90_SERVE_RMSNORM=0`` (checked per call) disables
the fast path entirely and every call runs the stock FlashInfer rmsnorm
dispatch unchanged.  Any call outside the guard (shape, dtype, layout,
architecture, token count, CUDA-graph or autotune constraints below) also
falls back to the stock path.

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

    _COMPILED = {}

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
        fw = cute.runtime.make_fake_tensor(
            dt, (h,), stride=(1,), assumed_align=align
        )
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


def _backend():
    """Requested backend: ``"cutedsl"`` (default), ``"cuda"``, or ``None``
    (kill switch).  Read per call so the kill switch works at runtime."""
    value = os.environ.get(_ENV_VAR, "").strip().lower()
    if value in ("0", "off", "false", "disable", "disabled"):
        return None
    if value == "cuda":
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
) -> bool:
    """Run the SM90 serve-shape RMSNorm when the call is inside the measured
    envelope.  Returns ``True`` when the output has been produced; ``False``
    means the caller must run the stock path (guard miss, kill switch,
    missing runtime, uncompiled variant under CUDA-graph capture, or
    autotune tuning mode)."""
    backend = _backend()
    if backend is None:
        return False
    if input.dim() != 2:
        return False
    tokens, hidden = input.shape
    if (
        input.dtype is not torch.bfloat16
        or weight.dtype is not torch.bfloat16
        or out.dtype is not torch.bfloat16
    ):
        return False
    if not 0 < tokens <= _MAX_TOKENS:
        return False
    if not _h_in_envelope(hidden, backend):
        return False
    if input.device.type != "cuda":
        return False
    if weight.shape != (hidden,) or out.shape != input.shape:
        return False
    if weight.device != input.device or out.device != input.device:
        return False
    if not (
        input.is_contiguous() and out.is_contiguous() and weight.is_contiguous()
    ):
        return False
    if (input.data_ptr() | out.data_ptr() | weight.data_ptr()) % 16 != 0:
        return False
    if torch.cuda.get_device_capability(input.device) != (9, 0):
        return False
    if _autotune_active():
        return False

    if backend == "cuda":
        if _cuda_module is None and torch.cuda.is_current_stream_capturing():
            # Never JIT-build the module under CUDA-graph capture.
            return False
        try:
            module = _get_cuda_module()
        except Exception:
            return False
        module.rmsnorm_serve_sm90(out, input, weight, float(eps))
        return True

    if not _HAS_CUTE_DSL:
        return False
    key = (hidden, input.dtype)
    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            # Never JIT-compile a new (H, dtype) variant under capture.
            return False
        try:
            fn = _build(hidden, input.dtype)
        except Exception:
            return False
        _COMPILED[key] = fn
    # The compiled kernel takes the TVM-FFI env stream, i.e. the caller's
    # current torch stream re-queried on every call (CUDA-graph safe).
    fn(input, weight, out, float(eps))
    return True


__all__ = ["try_rmsnorm_serve_sm90"]
