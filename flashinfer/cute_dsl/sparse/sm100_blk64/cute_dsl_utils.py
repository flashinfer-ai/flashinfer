# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# The ParamsBase, make_fake_tensor, and sub_packed_f32x2 implementations in
# this file are adapted from quack-kernels 0.4.1 (Apache-2.0) and modified for
# cudnn-frontend's CUTLASS DSL integration.

import inspect
from dataclasses import dataclass, fields
from functools import partial
from typing import Tuple, get_origin

import torch

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass.base_dsl.tvm_ffi_builder import spec
from cutlass.cutlass_dsl import NumericMeta
from cutlass.cute.runtime import from_dlpack

# Python scalars and CuTe numeric types are compile-time values. Everything
# else in a ParamsBase dataclass is flattened into MLIR values at JIT time.
_STATIC_TYPES = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


def _install_constexpr_tvm_ffi_converter() -> None:
    """Teach CUTLASS DSL's TVM-FFI converter about Constexpr annotations.

    Emitting ``ConstNone`` keeps static ``ParamsBase`` fields in the JIT
    specialization, including tuple and enum values that CUTLASS DSL 4.6's
    native Constexpr converter does not support. The NamedTuple case preserves
    its concrete field annotations when a broader tuple type is used at the
    call site.
    """
    import cutlass.cute._tvm_ffi_args_spec_converter as converter

    original = converter._convert_single_arg
    if getattr(original, "_cudnn_bsa_constexpr_compat", False):
        return
    supports_is_constexpr = (
        "is_constexpr" in inspect.signature(original).parameters
    )

    def convert_single_arg(
        arg,
        arg_name,
        arg_type,
        ctx,
        *,
        is_constexpr=False,
    ):
        if arg_type is not None and get_origin(arg_type) is cutlass.Constexpr:
            return spec.ConstNone(arg_name)
        if (
            isinstance(arg, tuple)
            and hasattr(type(arg), "_fields")
            and (arg_type is None or not hasattr(arg_type, "_fields"))
        ):
            arg_type = type(arg)
        if supports_is_constexpr:
            return original(
                arg,
                arg_name,
                arg_type,
                ctx,
                is_constexpr=is_constexpr,
            )
        return original(arg, arg_name, arg_type, ctx)

    convert_single_arg._cudnn_bsa_constexpr_compat = True
    converter._convert_single_arg = convert_single_arg


_install_constexpr_tvm_ffi_converter()

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}


def _partition_param_fields(obj):
    """Split dataclass fields into compile-time and MLIR-backed values."""
    all_fields = {field.name: getattr(obj, field.name) for field in fields(obj)}
    constexpr = {name: value for name, value in all_fields.items() if isinstance(value, _STATIC_TYPES)}
    dynamic = {name: value for name, value in all_fields.items() if not isinstance(value, _STATIC_TYPES)}
    return constexpr, dynamic


def _new_params_from_mlir_values(self, values):
    constexpr_fields, dynamic_fields = _partition_param_fields(self)
    values = list(values)
    for (name, field), num_values in zip(dynamic_fields.items(), self._values_pos):
        dynamic_fields[name] = cutlass.new_from_mlir_values(field, values[:num_values])
        values = values[num_values:]
    return self.__class__(**dynamic_fields, **constexpr_fields)


@dataclass
class ParamsBase:
    """Base class for CuTe DSL parameter dataclasses.

    Python scalar fields are JIT-specialized. Tensor and CuTe scalar fields
    are flattened to MLIR values and rebuilt inside the compiled function.
    """

    def __extract_mlir_values__(self):
        _, dynamic_fields = _partition_param_fields(self)
        values, self._values_pos = [], []
        for obj in dynamic_fields.values():
            obj_values = cutlass.extract_mlir_values(obj)
            values.extend(obj_values)
            self._values_pos.append(len(obj_values))
        return values

    __new_from_mlir_values__ = _new_params_from_mlir_values


def make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1):
    """Create a fake compact tensor with symbolic non-leading strides."""
    if dtype is None:
        return None
    if leading_dim < 0:
        leading_dim += len(shape)
    stride = tuple(1 if dim == leading_dim else cute.sym_int64(divisibility=divisibility) for dim in range(len(shape)))
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=stride,
        assumed_align=divisibility * dtype.width // 8,
    )


sub_packed_f32x2 = partial(
    cute.arch.calc_packed_f32x2_op,
    src_c=None,
    calc_func=nvvm.sub_packed_f32x2,
)


def assume_strides_aligned(t):
    """Assume all strides except the last are divisible by 128 bits.

    Python int strides (e.g., stride=0 from GQA expand) are kept as-is
    since they're static and don't need alignment assumptions.
    """
    divby = 128 // t.element_type.width
    strides = tuple(s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1])
    return (*strides, t.stride[-1])


def assume_tensor_aligned(t):
    """Rebuild a tensor with 128-bit aligned stride assumptions. Passes through None."""
    if t is None:
        return None
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t)))


def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=True):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1."""
    # NOTE: torch 2.9.1 doesn't support fp8 via DLPack but 2.11.0 nightly does
    # currently export raw bytes as uint8 and tell cutlass correct type
    # can directly export as fp8 when torch supports it
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = from_dlpack(
            t.view(torch.uint8).detach(),
            assumed_align=assumed_align,
            enable_tvm_ffi=enable_tvm_ffi,
        )
        tensor.element_type = cutlass.Float8E4M3FN if t.dtype == torch.float8_e4m3fn else cutlass.Float8E5M2
    else:
        tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


def get_broadcast_dims(tensor: torch.Tensor) -> Tuple[bool, ...]:
    """Return tuple of bools indicating which dims have stride=0 (broadcast).

    This is useful for compile keys since CuTe's mark_layout_dynamic() keeps
    stride=0 as static, meaning kernels compiled with different broadcast
    patterns are not interchangeable.
    """
    return tuple(s == 0 for s in tensor.stride())
