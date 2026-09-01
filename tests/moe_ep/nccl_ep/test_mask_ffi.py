"""ctypes shim unit tests — no libnccl_ep.so required.

The shim is driven with a fake "library" object whose attributes are
callables carrying settable ``argtypes``/``restype``, which is all ctypes
asks of them. That lets us assert the marshalling contract (device vs host
pointer, explicit argtypes) on any host.
"""

from __future__ import annotations

import types
from ctypes import POINTER, c_int, c_void_p

import pytest

from flashinfer.moe_ep.backends.split.comm.nccl_ep._mask_ffi import (
    _NCCL_RESULT_NAMES,
    _SIGS,
    _MaskFfi,
)
from flashinfer.moe_ep.errors import (
    MoEEpFaultToleranceUnsupportedError,
    MoEEpTransportError,
)


class _FakeFn:
    def __init__(self, rc=0, on_call=None):
        self.rc = rc
        self.on_call = on_call
        self.calls: list[tuple] = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls.append(args)
        if self.on_call is not None:
            self.on_call(*args)
        return self.rc


def _fake_lib(rc=0, omit=(), on_call=None):
    ns = types.SimpleNamespace()
    for name in _SIGS:
        if name in omit:
            continue
        setattr(ns, name, _FakeFn(rc=rc, on_call=on_call))
    return ns


class _FakeGroup:
    """Stands in for nccl.ep.Group — only ``.ptr`` is used by the shim."""

    def __init__(self, ptr=0xDEADBEEF):
        self.ptr = ptr


class TestSymbolBinding:
    def test_binds_all_signatures_exactly(self):
        lib = _fake_lib()
        ffi = _MaskFfi(lib=lib)
        assert ffi.available is True
        assert ffi.missing == ()
        # Guards the classic ctypes bug: without explicit argtypes a 64-bit
        # pointer is silently truncated to int on the way in.
        for name, (argtypes, restype) in _SIGS.items():
            fn = getattr(lib, name)
            assert fn.argtypes == argtypes, name
            assert fn.restype is restype, name

    def test_pointer_args_are_void_p(self):
        # All three group/pointer/stream args must be c_void_p, and the
        # async-error out-param must be a POINTER(c_int).
        assert _SIGS["ncclEpMaskQuery"][0] == [c_void_p, c_void_p, c_void_p]
        assert _SIGS["ncclEpGetAsyncError"][0] == [c_void_p, POINTER(c_int)]

    def test_missing_symbol_reports_unavailable(self):
        ffi = _MaskFfi(lib=_fake_lib(omit=("ncclEpMaskClean",)))
        assert ffi.available is False
        assert ffi.missing == ("ncclEpMaskClean",)

    def test_no_library_degrades_quietly(self):
        # A host with no NCCL at all must not blow up at construction —
        # supports_fault_tolerance() calls this unconditionally.
        # lib=None triggers real resolution; depending on the host it may or
        # may not find libnccl_ep. What matters is that it did not raise.
        ffi = _MaskFfi()
        assert isinstance(ffi.available, bool)
        assert isinstance(ffi.missing, tuple)


class TestMarshalling:
    def test_query_passes_device_pointer(self):
        lib = _fake_lib()
        ffi = _MaskFfi(lib=lib)
        ffi.mask_query(_FakeGroup(0x1000), dev_ptr=0x2000, stream=0x3000)
        (args,) = lib.ncclEpMaskQuery.calls
        assert [a.value for a in args] == [0x1000, 0x2000, 0x3000]

    def test_update_passes_host_pointer(self):
        # Update takes a HOST pointer where Query takes a DEVICE one; the
        # shim must not "helpfully" unify them.
        lib = _fake_lib()
        ffi = _MaskFfi(lib=lib)
        ffi.mask_update(_FakeGroup(0x1000), host_ptr=0x4000, stream=0x3000)
        (args,) = lib.ncclEpMaskUpdate.calls
        assert [a.value for a in args] == [0x1000, 0x4000, 0x3000]

    def test_clean_passes_group_and_stream(self):
        lib = _fake_lib()
        _MaskFfi(lib=lib).mask_clean(_FakeGroup(0x1000), stream=0x3000)
        (args,) = lib.ncclEpMaskClean.calls
        assert [a.value for a in args] == [0x1000, 0x3000]

    def test_get_async_error_marshals_out_param(self):
        def set_flag(_group, out_ref):
            out_ref._obj.value = 1

        lib = _fake_lib(on_call=set_flag)
        assert _MaskFfi(lib=lib).get_async_error(_FakeGroup()) is True

    def test_get_async_error_false_when_unset(self):
        assert _MaskFfi(lib=_fake_lib()).get_async_error(_FakeGroup()) is False

    def test_accepts_raw_address(self):
        lib = _fake_lib()
        _MaskFfi(lib=lib).error_clear(0x99)
        (args,) = lib.ncclEpErrorClear.calls
        assert args[0].value == 0x99


class TestErrorMapping:
    def test_invalid_usage_carries_enable_mask_hint(self):
        ffi = _MaskFfi(lib=_fake_lib(rc=5))
        with pytest.raises(MoEEpTransportError) as ei:
            ffi.mask_query(_FakeGroup(), 0x1, 0x2)
        msg = str(ei.value)
        assert "ncclInvalidUsage (5)" in msg
        assert "enable_mask" in msg
        assert ei.value.code == 5

    def test_other_codes_map_by_name(self):
        ffi = _MaskFfi(lib=_fake_lib(rc=3))
        with pytest.raises(MoEEpTransportError, match="ncclInternalError"):
            ffi.mask_clean(_FakeGroup(), 0x2)

    def test_unknown_code_still_raises(self):
        ffi = _MaskFfi(lib=_fake_lib(rc=99))
        with pytest.raises(MoEEpTransportError) as ei:
            ffi.error_clear(_FakeGroup())
        assert ei.value.code == 99

    def test_calling_a_missing_symbol_is_actionable(self):
        ffi = _MaskFfi(lib=_fake_lib(omit=("ncclEpMaskClean",)))
        with pytest.raises(MoEEpFaultToleranceUnsupportedError) as ei:
            ffi.mask_clean(_FakeGroup(), 0x2)
        assert "ncclEpMaskClean" in str(ei.value)
        assert "nccl4py" in str(ei.value)

    def test_result_names_cover_the_nccl_enum(self):
        assert _NCCL_RESULT_NAMES[0] == "ncclSuccess"
        assert _NCCL_RESULT_NAMES[5] == "ncclInvalidUsage"


class TestNativeFirst:
    """The migration hook: a native Group method wins over ctypes."""

    def test_native_query_preferred(self):
        lib = _fake_lib()
        group = _FakeGroup()
        seen = {}
        group.mask_query = lambda dev_ptr, stream: seen.update(
            dev=dev_ptr, stream=stream
        )
        _MaskFfi(lib=lib).mask_query(group, dev_ptr=0x7, stream=0x8)
        assert seen == {"dev": 0x7, "stream": 0x8}
        assert lib.ncclEpMaskQuery.calls == []  # ctypes path untouched

    def test_native_async_error_preferred(self):
        lib = _fake_lib()
        group = _FakeGroup()
        group.get_async_error = lambda: 1
        assert _MaskFfi(lib=lib).get_async_error(group) is True
        assert lib.ncclEpGetAsyncError.calls == []
