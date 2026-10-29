# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import vector
from cutlass._mlir.extras import types as T_
from cutlass._mlir import ir as _ir


def _ir_elem_ty(dtype):
    if dtype is cutlass.Float32:
        return T_.f32()
    if dtype is cutlass.Int32:
        return T_.i32()
    if dtype is cutlass.Float16:
        return T_.f16()
    if dtype is cutlass.BFloat16:
        return T_.bf16()
    raise TypeError(f"RegTile: unsupported dtype {dtype!r}")


def vec_slice(vec, start: int, length: int):
    n = int(vec.shape[0])
    assert 0 <= start and start + length <= n, f"vec_slice: [{start}:{start + length}) out of range [0:{n})"
    elem_ty = _ir_elem_ty(vec.dtype)
    out_ty = _ir.VectorType.get([length], elem_ty)
    sliced = vector.extract_strided_slice(
        out_ty,
        vec.ir_value(),
        offsets=[start],
        sizes=[length],
        strides=[1],
    )
    return cutlass.Vector(sliced, dtype=vec.dtype)


def _vec_width(vec):
    return int(_ir.VectorType(vec.ir_value().type).shape[0])


def vec_concat(parts):
    assert len(parts) >= 1, "vec_concat: need at least one part"
    dtype = parts[0].dtype
    widths = [_vec_width(p) for p in parts]
    total = sum(widths)
    elem_ty = _ir_elem_ty(dtype)
    out_ty = _ir.VectorType.get([total], elem_ty)

    acc = vector.broadcast(out_ty, _zero_scalar(dtype))
    offset = 0
    for p, w in zip(parts, widths):
        acc = vector.insert_strided_slice(
            p.ir_value(),
            acc,
            offsets=[offset],
            strides=[1],
        )
        offset += w
    return cutlass.Vector(acc, dtype=dtype)


def _zero_scalar(dtype):
    if dtype is cutlass.Float32:
        return cutlass.Float32(0.0).ir_value()
    if dtype is cutlass.Int32:
        return cutlass.Int32(0).ir_value()
    if dtype is cutlass.Float16:
        return cutlass.Float16(0.0).ir_value()
    if dtype is cutlass.BFloat16:
        return cutlass.BFloat16(0.0).ir_value()
    raise TypeError(f"_zero_scalar: unsupported dtype {dtype!r}")


class RegTile:

    __slots__ = ("_vec", "_size")

    def __init__(self, vec_or_n, dtype=None, *, size: int = None):
        if isinstance(vec_or_n, int):
            assert dtype is not None, "RegTile(int, ...) requires explicit dtype"
            self._vec = cutlass.Vector.from_elements(
                tuple(_zero_py(dtype) for _ in range(vec_or_n)),
                dtype,
            )
            self._size = vec_or_n
        else:
            self._vec = vec_or_n
            if size is not None:
                self._size = int(size)
            else:
                self._size = _vec_width(vec_or_n)

    @classmethod
    def empty(cls, num_elems: int, dtype):
        return cls(num_elems, dtype=dtype)

    @property
    def vec(self):
        return self._vec

    @property
    def dtype(self):
        return self._vec.dtype

    def __len__(self):
        return self._size

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._size)
            assert step == 1, f"RegTile slice step must be 1, got {step}"
            return RegTile(vec_slice(self._vec, start, stop - start), size=stop - start)
        return self._vec[key]

    def to(self, dtype):
        return RegTile(self._vec.to(dtype), size=self._size)

    def __mul__(self, other):
        rhs = other.vec if isinstance(other, RegTile) else other
        return RegTile(self._vec * rhs, size=self._size)

    def __rmul__(self, other):
        rhs = other.vec if isinstance(other, RegTile) else other
        return RegTile(rhs * self._vec, size=self._size)

    def __sub__(self, other):
        rhs = other.vec if isinstance(other, RegTile) else other
        return RegTile(self._vec - rhs, size=self._size)

    def __rsub__(self, other):
        rhs = other.vec if isinstance(other, RegTile) else other
        return RegTile(rhs - self._vec, size=self._size)

    def __add__(self, other):
        rhs = other.vec if isinstance(other, RegTile) else other
        return RegTile(self._vec + rhs, size=self._size)

    def __radd__(self, other):
        rhs = other.vec if isinstance(other, RegTile) else other
        return RegTile(rhs + self._vec, size=self._size)


def _zero_py(dtype):
    if dtype is cutlass.Float32:
        return cutlass.Float32(0.0)
    if dtype is cutlass.Int32:
        return cutlass.Int32(0)
    if dtype is cutlass.Float16:
        return cutlass.Float16(0.0)
    if dtype is cutlass.BFloat16:
        return cutlass.BFloat16(0.0)
    raise TypeError(f"_zero_py: unsupported dtype {dtype!r}")
