"""ctypes shim for NCCL-EP's fault-tolerance mask API.

``nccl-extensions`` binds ``GroupConfig.enable_mask`` and ``timeout_ns`` (so
masking can be switched on) but its ``Group`` class stops at ``create`` /
``create_handle`` / ``destroy`` / ``.ptr`` — the five mask functions are not
exposed to Python:

    ncclEpMaskQuery / ncclEpMaskUpdate / ncclEpMaskClean
    ncclEpGetAsyncError / ncclEpErrorClear

``Group.ptr`` is the raw ``ncclEpGroup_t``, so we can call them directly until
nccl-extensions catches up. Every method here tries a native ``Group`` method FIRST
and only then falls back to ctypes, so the shim retires itself with no
call-site churn the day those bindings land.

Which library: the symbols live in **libnccl_ep.so**, not libnccl.so.2.
``nccl.ep`` dlopens it with ``RTLD_GLOBAL`` and resolves through
``dlsym(RTLD_DEFAULT, ...)``, so once ``nccl.ep`` has initialized, the symbols
are in the process-global namespace and ``CDLL(None)`` binds *exactly* the
library the caller's group came from. That matters: resolving a path
ourselves could bind a second, different libnccl_ep if the wheel and
LD_LIBRARY_PATH disagree, and calling into it with a group created by the
other one is undefined behaviour. Path resolution is only a fallback for
probing before any group exists.

TODO(moe_ep): drop the ctypes fallback once nccl-extensions binds ncclEpMask* on
``nccl.ep.Group``; the native-first dispatch below already prefers it.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
from ctypes import POINTER, byref, c_int, c_void_p
from pathlib import Path

from .....errors import MoEEpFaultToleranceUnsupportedError, MoEEpTransportError

# ncclEpGroup_t and cudaStream_t are opaque pointers; ncclResult_t is an enum.
# Declaring these explicitly is not optional: without argtypes, ctypes
# int-truncates 64-bit pointers on the way in.
_SIGS: dict[str, tuple[list, type]] = {
    # group, int* DEVICE [nRanks] (1 = active), stream
    "ncclEpMaskQuery": ([c_void_p, c_void_p, c_void_p], c_int),
    # group, const int* HOST [nRanks] (1 = active), stream  <- host, unlike Query
    "ncclEpMaskUpdate": ([c_void_p, c_void_p, c_void_p], c_int),
    # group, stream
    "ncclEpMaskClean": ([c_void_p, c_void_p], c_int),
    "ncclEpGetAsyncError": ([c_void_p, POINTER(c_int)], c_int),
    "ncclEpErrorClear": ([c_void_p], c_int),
}

_NCCL_RESULT_NAMES = {
    0: "ncclSuccess",
    1: "ncclUnhandledCudaError",
    2: "ncclSystemError",
    3: "ncclInternalError",
    4: "ncclInvalidArgument",
    5: "ncclInvalidUsage",
    6: "ncclRemoteError",
    7: "ncclInProgress",
}

_UPGRADE_HINT = (
    "Fault tolerance needs an NCCL-EP build carrying the ncclEpMask* API "
    "(nccl-ep >= v0.1.0 with active-mask support). Upgrade the nccl-extensions "
    "wheel that ships libnccl_ep.so, and make sure it is the one actually "
    "loaded (check `python -c 'import nccl.ep; print(nccl.ep.get_lib_path(), "
    "nccl.ep.get_lib_version())'`)."
)


def _resolve_libnccl_ep() -> ctypes.CDLL | None:
    """Return a handle exporting the mask symbols, or None.

    Mirrors nccl-extensions' own resolution order so we bind the same library
    it does. The global namespace is tried first because that is the one case
    where the identity is *guaranteed* rather than merely likely.
    """
    # 1. Process-global namespace. If nccl.ep has initialized, its
    #    RTLD_GLOBAL dlopen already put the symbols here.
    try:
        main = ctypes.CDLL(None)
        main.ncclEpMaskQuery  # noqa: B018 - presence probe
        return main
    except (OSError, AttributeError):
        pass

    # 2. Ask nccl.ep itself, then fall back to its package layout, then to
    #    the linker's own search. ``get_lib_path()`` reports the library the
    #    bindings actually loaded, which is the only answer guaranteed to
    #    match the one a Group would come from; it raises rather than returns
    #    None when libnccl.so.2 is missing, hence the broad except.
    #
    #    Layout note: since nccl-extensions 0.1.0 the bundled libraries are
    #    split per CUDA major (nccl/ep/lib/cu{12,13}/libnccl_ep.so), whereas
    #    nccl4py <= 0.3.x shipped a flat nccl/ep/lib/libnccl_ep.so. Probe the
    #    CUDA-matched directory first, then the other, then the flat legacy
    #    path. libnccl_ep.so has NEEDED libnccl.so.2, so preload that first
    #    exactly as the transport backend does.
    candidates: list[str] = []
    try:
        import nccl.ep as _nccl_ep  # type: ignore[import-not-found]

        with contextlib.suppress(Exception):
            lib_path = _nccl_ep.get_lib_path()
            if lib_path is not None:
                candidates.append(str(lib_path))

        ep_lib = Path(_nccl_ep.__path__[0]) / "lib"
        majors = ["12", "13"]
        with contextlib.suppress(Exception):
            from nccl._extensions._runtime import (  # type: ignore[import-not-found]
                cuda_major,
            )

            majors.remove(str(cuda_major()))
            majors.insert(0, str(cuda_major()))
        candidates += [str(ep_lib / f"cu{m}" / "libnccl_ep.so") for m in majors]
        candidates.append(str(ep_lib / "libnccl_ep.so"))
    except Exception:
        pass
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        candidates += [
            str(Path(conda) / sub / "libnccl_ep.so") for sub in ("lib", "lib64")
        ]
    candidates.append("libnccl_ep.so")  # SONAME fallback

    from . import _preload_libnccl

    for cand in candidates:
        if cand != "libnccl_ep.so" and not Path(cand).exists():
            continue
        # libnccl.so.2 is a NEEDED of libnccl_ep.so; a failure here is not
        # fatal (the linker may still find it) so the CDLL below decides.
        with contextlib.suppress(Exception):
            _preload_libnccl()
        try:
            return ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
    return None


class _MaskFfi:
    """Bound ``ncclEpMask*`` entry points, or a clear account of what is missing.

    Never raises at construction: a host with no NCCL at all must be able to
    call ``supports_fault_tolerance("nccl_ep")`` and simply get False.
    """

    def __init__(self, lib=None) -> None:
        self._fns: dict = {}
        if lib is None:
            lib = _resolve_libnccl_ep()
        if lib is None:
            self.missing = tuple(_SIGS)
            self.available = False
            return
        missing: list[str] = []
        for name, (argtypes, restype) in _SIGS.items():
            try:
                fn = getattr(lib, name)
            except AttributeError:
                missing.append(name)
                continue
            fn.argtypes = argtypes
            fn.restype = restype
            self._fns[name] = fn
        self.missing = tuple(missing)
        self.available = not missing

    # ------------------------------------------------------------------ core

    def _call(self, name: str, *args) -> None:
        fn = self._fns.get(name)
        if fn is None:
            raise MoEEpFaultToleranceUnsupportedError(
                f"{name} is not exported by the loaded libnccl_ep "
                f"(missing: {', '.join(self.missing) or name}). {_UPGRADE_HINT}"
            )
        rc = fn(*args)
        if rc != 0:
            detail = ""
            if rc == 5:  # ncclInvalidUsage — by far the likeliest misuse
                detail = (
                    "  (the EP group was created without enable_mask; pass "
                    "FleetAlgoKnobFaultTolerance() at Fleet construction)"
                )
            raise MoEEpTransportError(
                name, rc, detail, code_name=_NCCL_RESULT_NAMES.get(rc)
            )

    @staticmethod
    def _ptr(group) -> int:
        """Accept either an ``nccl.ep.Group`` or a raw address."""
        return int(getattr(group, "ptr", group))

    # ------------------------------------------------------- public wrappers
    #
    # Each prefers a native nccl.ep Group method when one exists, so this
    # module becomes dead weight (not a blocker) once those land.

    def mask_query(self, group, dev_ptr: int, stream: int) -> None:
        native = getattr(group, "mask_query", None)
        if native is not None:
            native(dev_ptr, stream=stream)
            return
        self._call(
            "ncclEpMaskQuery",
            c_void_p(self._ptr(group)),
            c_void_p(dev_ptr),
            c_void_p(stream),
        )

    def mask_update(self, group, host_ptr: int, stream: int) -> None:
        native = getattr(group, "mask_update", None)
        if native is not None:
            native(host_ptr, stream=stream)
            return
        self._call(
            "ncclEpMaskUpdate",
            c_void_p(self._ptr(group)),
            c_void_p(host_ptr),
            c_void_p(stream),
        )

    def mask_clean(self, group, stream: int) -> None:
        native = getattr(group, "mask_clean", None)
        if native is not None:
            native(stream=stream)
            return
        self._call("ncclEpMaskClean", c_void_p(self._ptr(group)), c_void_p(stream))

    def get_async_error(self, group) -> bool:
        native = getattr(group, "get_async_error", None)
        if native is not None:
            return bool(native())
        out = c_int()
        self._call("ncclEpGetAsyncError", c_void_p(self._ptr(group)), byref(out))
        return bool(out.value)

    def error_clear(self, group) -> None:
        native = getattr(group, "error_clear", None)
        if native is not None:
            native()
            return
        self._call("ncclEpErrorClear", c_void_p(self._ptr(group)))


@functools.cache
def mask_ffi() -> _MaskFfi:
    """Process-wide shim instance (symbol binding is done once)."""
    return _MaskFfi()


__all__ = ["mask_ffi", "_MaskFfi"]
