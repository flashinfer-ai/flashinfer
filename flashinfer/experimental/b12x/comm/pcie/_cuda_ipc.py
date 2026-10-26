"""Minimal CUDA runtime IPC wrapper used by distributed integration code."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch  # noqa: F401


cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int
cudaStream_t = ctypes.c_void_p


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


@dataclass(frozen=True)
class _Function:
    name: str
    restype: Any
    argtypes: list[Any]


def _find_loaded_library(lib_name: str) -> Optional[str]:
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                start = line.index("/")
                return line[start:].strip()
    return None


class CudaRTLibrary:
    _FUNCTIONS = [
        _Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),
        _Function("cudaSetDevice", cudaError_t, [ctypes.c_int]),
        _Function(
            "cudaMalloc",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        ),
        _Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        _Function(
            "cudaMemset", cudaError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        ),
        _Function(
            "cudaMemcpyAsync",
            cudaError_t,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                cudaMemcpyKind,
                cudaStream_t,
            ],
        ),
        _Function(
            "cudaIpcGetMemHandle",
            cudaError_t,
            [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p],
        ),
        _Function(
            "cudaIpcOpenMemHandle",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint],
        ),
        _Function("cudaIpcCloseMemHandle", cudaError_t, [ctypes.c_void_p]),
    ]

    _LIB_CACHE: Dict[str, Any] = {}
    _FUNC_CACHE: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = _find_loaded_library("libcudart")
            if so_file is None:
                raise RuntimeError("libcudart is not loaded in the current process")
        if so_file not in self._LIB_CACHE:
            self._LIB_CACHE[so_file] = ctypes.CDLL(so_file)
        self.lib = self._LIB_CACHE[so_file]
        if so_file not in self._FUNC_CACHE:
            funcs: Dict[str, Any] = {}
            for spec in self._FUNCTIONS:
                func = getattr(self.lib, spec.name)
                func.restype = spec.restype
                func.argtypes = spec.argtypes
                funcs[spec.name] = func
            self._FUNC_CACHE[so_file] = funcs
        self.funcs = self._FUNC_CACHE[so_file]

    def _check(self, result: int) -> None:
        if result != 0:
            error = self.funcs["cudaGetErrorString"](result).decode("utf-8")
            raise RuntimeError(f"CUDART error: {error}")

    @staticmethod
    def _void_p(ptr: int | ctypes.c_void_p) -> ctypes.c_void_p:
        if isinstance(ptr, ctypes.c_void_p):
            return ptr
        return ctypes.c_void_p(int(ptr))

    def cudaSetDevice(self, device: int) -> None:
        self._check(self.funcs["cudaSetDevice"](device))

    def cudaMalloc(self, size: int) -> int:
        ptr = ctypes.c_void_p()
        self._check(self.funcs["cudaMalloc"](ctypes.byref(ptr), size))
        return int(ptr.value)

    def cudaFree(self, ptr: int | ctypes.c_void_p) -> None:
        self._check(self.funcs["cudaFree"](self._void_p(ptr)))

    def cudaMemset(self, ptr: int | ctypes.c_void_p, value: int, count: int) -> None:
        self._check(self.funcs["cudaMemset"](self._void_p(ptr), value, count))

    def cudaMemcpyAsync(self, dst: int, src: int, count: int, stream: int) -> None:
        cudaMemcpyDefault = 4
        self._check(
            self.funcs["cudaMemcpyAsync"](
                self._void_p(dst),
                self._void_p(src),
                count,
                cudaMemcpyDefault,
                cudaStream_t(int(stream)),
            )
        )

    def cudaIpcGetMemHandleBytes(self, ptr: int | ctypes.c_void_p) -> bytes:
        handle = cudaIpcMemHandle_t()
        self._check(
            self.funcs["cudaIpcGetMemHandle"](ctypes.byref(handle), self._void_p(ptr))
        )
        return bytes(handle)

    def cudaIpcOpenMemHandleBytes(self, handle_bytes: bytes) -> int:
        if len(handle_bytes) != ctypes.sizeof(cudaIpcMemHandle_t):
            raise ValueError("invalid CUDA IPC handle size")
        handle = cudaIpcMemHandle_t.from_buffer_copy(handle_bytes)
        cudaIpcMemLazyEnablePeerAccess = 1
        ptr = ctypes.c_void_p()
        self._check(
            self.funcs["cudaIpcOpenMemHandle"](
                ctypes.byref(ptr), handle, cudaIpcMemLazyEnablePeerAccess
            )
        )
        return int(ptr.value)

    def cudaIpcCloseMemHandle(self, ptr: int | ctypes.c_void_p) -> None:
        self._check(self.funcs["cudaIpcCloseMemHandle"](self._void_p(ptr)))
