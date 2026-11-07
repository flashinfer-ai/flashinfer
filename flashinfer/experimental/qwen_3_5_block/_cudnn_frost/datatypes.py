# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import importlib


def is_windows():
    return sys.platform.startswith("win")


module_name = ".Release._compiled_module" if is_windows() else "._compiled_module"

from . import _compiled_module as _pybind_module  # [vendor.py P2] the shim, not the nanobind extension

globals()["cudnn_data_type"] = getattr(_pybind_module, "data_type")

torch_available = None
_torch_to_cudnn_data_type_dict = None

# Optional CUTLASS integration
cutlass_available = None
_torch_to_cutlass_data_type_dict = None


def is_torch_available():
    global torch_available, _torch_to_cudnn_data_type_dict
    # this condition ensures that datatype mapping is only created once
    if torch_available is None:
        try:
            import torch

            torch_available = True
            _torch_to_cudnn_data_type_dict = {
                torch.half: cudnn_data_type.HALF,
                torch.float16: cudnn_data_type.HALF,
                torch.bfloat16: cudnn_data_type.BFLOAT16,
                torch.float: cudnn_data_type.FLOAT,
                torch.float32: cudnn_data_type.FLOAT,
                torch.double: cudnn_data_type.DOUBLE,
                torch.float64: cudnn_data_type.DOUBLE,
                torch.int8: cudnn_data_type.INT8,
                torch.int32: cudnn_data_type.INT32,
                torch.int64: cudnn_data_type.INT64,
                torch.uint8: cudnn_data_type.UINT8,
                torch.bool: cudnn_data_type.BOOLEAN,
            }

            def possibly_add_type(torch_type_name, cudnn_type):
                # Only try adding the type if the version of torch being used supports it
                if hasattr(torch, torch_type_name):
                    torch_type = getattr(torch, torch_type_name)
                    _torch_to_cudnn_data_type_dict[torch_type] = cudnn_type

            possibly_add_type("float8_e4m3fn", cudnn_data_type.FP8_E4M3)
            possibly_add_type("float8_e5m2", cudnn_data_type.FP8_E5M2)

            possibly_add_type("float8_e8m0fnu", cudnn_data_type.FP8_E8M0)
            possibly_add_type("float4_e2m1fn_x2", cudnn_data_type.FP4_E2M1)

        except ImportError:
            torch_available = False
            _torch_to_cudnn_data_type_dict = {}
    return torch_available


def is_cutlass_available():
    global cutlass_available
    if cutlass_available is None:
        try:
            import cutlass

            cutlass_available = True
        except ImportError:
            cutlass_available = False
    return cutlass_available


def _is_torch_to_cutlass_available():
    global _torch_to_cutlass_data_type_dict
    if _torch_to_cutlass_data_type_dict is None:
        try:
            import torch
            import cutlass

            mapping = {
                torch.half: getattr(cutlass, "Float16", None),
                getattr(torch, "float16", torch.half): getattr(cutlass, "Float16", None),
                getattr(torch, "bfloat16", None): getattr(cutlass, "BFloat16", None),
                torch.float: getattr(cutlass, "Float32", None),
                getattr(torch, "float32", torch.float): getattr(cutlass, "Float32", None),
                torch.double: getattr(cutlass, "Float64", None),
                getattr(torch, "float64", torch.double): getattr(cutlass, "Float64", None),
                getattr(torch, "int8", None): getattr(cutlass, "Int8", None),
                getattr(torch, "int32", None): getattr(cutlass, "Int32", None),
                getattr(torch, "int64", None): getattr(cutlass, "Int64", None),
                getattr(torch, "uint8", None): getattr(cutlass, "Uint8", None),
                getattr(torch, "bool", None): getattr(cutlass, "Boolean", None),
                getattr(torch, "float8_e4m3fn", None): getattr(cutlass, "Float8E4M3FN", None),
                getattr(torch, "float8_e5m2", None): getattr(cutlass, "Float8E5M2", None),
                getattr(torch, "float8_e8m0fnu", None): getattr(cutlass, "Float8E8M0FNU", None),
                getattr(torch, "float4_e2m1fn_x2", None): getattr(cutlass, "Float4E2M1FN", None),
            }
            _torch_to_cutlass_data_type_dict = {t: c for t, c in mapping.items() if t is not None and c is not None}
        except ImportError:
            _torch_to_cutlass_data_type_dict = {}
    return bool(_torch_to_cutlass_data_type_dict)


# Framework-neutral dtype-name -> cutlass mapping. Keyed on np.dtype(x).name so
# numpy, ml_dtypes, and string dtypes all resolve without importing torch or ml_dtypes.
_dtype_name_to_cutlass_data_type_dict = None


def _get_dtype_name_to_cutlass_dict():
    global _dtype_name_to_cutlass_data_type_dict
    if _dtype_name_to_cutlass_data_type_dict is None:
        import cutlass

        names = {
            "float16": "Float16",
            "bfloat16": "BFloat16",
            "float32": "Float32",
            "float64": "Float64",
            "uint8": "Uint8",
            "int8": "Int8",
            "int32": "Int32",
            "int64": "Int64",
            "bool": "Boolean",
            "float8_e4m3fn": "Float8E4M3FN",
            "float8_e5m2": "Float8E5M2",
            "float8_e8m0fnu": "Float8E8M0FNU",
            "float4_e2m1fn": "Float4E2M1FN",
        }
        _dtype_name_to_cutlass_data_type_dict = {name: getattr(cutlass, attr) for name, attr in names.items() if getattr(cutlass, attr, None) is not None}
    return _dtype_name_to_cutlass_data_type_dict


def _dtype_name_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2: bool = False):
    """Map a numpy/ml_dtypes dtype (or dtype name string) to a cutlass type, or None."""
    try:
        import numpy as np

        name = np.dtype(data_type).name
    except Exception:
        try:
            # dtype names like "bfloat16"/"float8_e4m3fn" only resolve once
            # ml_dtypes has registered its numpy extension types.
            import ml_dtypes
            import numpy as np

            name = np.dtype(data_type).name
        except Exception:
            return None
    if interpret_uint8_as_fp4x2 and name == "uint8":
        import cutlass

        return getattr(cutlass, "Float4E2M1FN", None)
    return _get_dtype_name_to_cutlass_dict().get(name, None)


def torch_to_cudnn_data_type(torch_data_type) -> cudnn_data_type:
    """The ``cudnn.data_type`` of a ``torch.dtype`` (``torch.bfloat16`` -> ``BFLOAT16``, ``torch.float8_e4m3fn`` ->
    ``FP8_E4M3``, ...), or None when torch is not installed or the dtype has no cuDNN twin.

    The public spelling of the mapping an integration reads to declare graph tensors from torch buffers; the private
    name below is kept as an alias of it, so a caller written against that spelling keeps working."""
    if is_torch_available():
        return _torch_to_cudnn_data_type_dict.get(torch_data_type, None)
    return None


# Kept: a private name read outside this package is still a caller boundary (AGENTS.md Rule 9).
_torch_to_cudnn_data_type = torch_to_cudnn_data_type


# cuDNN enum -> the dtype NAME frost.buffers speaks (its DTYPES table is keyed
# by name because a DLPack view needs no tensor library). Built once: this is
# read per operand per execute.
_CUDNN_TO_FROST_DTYPE_NAME = {
    cudnn_data_type.FLOAT: "float32",
    cudnn_data_type.HALF: "float16",
    cudnn_data_type.BFLOAT16: "bfloat16",
    cudnn_data_type.DOUBLE: "float64",
    cudnn_data_type.INT64: "int64",
    cudnn_data_type.INT32: "int32",
    cudnn_data_type.INT8: "int8",
    cudnn_data_type.UINT8: "uint8",
    cudnn_data_type.BOOLEAN: "bool",
    cudnn_data_type.FP8_E4M3: "float8_e4m3fn",
    cudnn_data_type.FP8_E5M2: "float8_e5m2",
    cudnn_data_type.FP8_E8M0: "float8_e8m0fnu",
}


def _cudnn_to_frost_dtype_name(data_type):
    """Name for a cuDNN dtype in the vocabulary ``frost.buffers.DTYPES`` uses,
    or None when the type has no DLPack-expressible name (the sub-byte ones —
    fp4 has a DLPack code but an itemsize of 0 bytes, so a caller must pass it
    as a typed buffer rather than as a bare address).

    Lives here so the mapping has one home; frost imports it rather than
    keeping a second table."""
    return _CUDNN_TO_FROST_DTYPE_NAME.get(data_type)


_buffer_dtype_to_cudnn_dict = None


def _buffer_dtype_to_cudnn(dtype) -> cudnn_data_type:
    """cuDNN enum for however a caller's buffer spells its dtype, or None.

    ONE table for every framework rather than one per framework: a torch dtype,
    a numpy/cupy dtype and a bare name string are all hashable and mutually
    unequal, so they coexist as keys and the caller needs no branch. numpy has
    no bfloat16, which is why the name keys exist at all — that is the dtype a
    DLPack read hands back for the case torch's ``__cuda_array_interface__``
    cannot express.
    """
    global _buffer_dtype_to_cudnn_dict
    if _buffer_dtype_to_cudnn_dict is None:
        table = {name: enum for enum, name in _CUDNN_TO_FROST_DTYPE_NAME.items()}
        if is_torch_available():
            table.update(_torch_to_cudnn_data_type_dict)
        try:
            import numpy
        except ImportError:
            pass
        else:
            for name, enum in list(table.items()):
                if isinstance(name, str):
                    try:
                        table[numpy.dtype(name)] = enum
                    except TypeError:
                        pass  # no numpy spelling (bfloat16); the name key serves it
        _buffer_dtype_to_cudnn_dict = table
    return _buffer_dtype_to_cudnn_dict.get(dtype)


def _torch_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2: bool = False):
    # A torch dtype can only be passed in if torch is already imported, so probing
    # sys.modules avoids importing torch on behalf of other frameworks' dtypes.
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(data_type, torch.dtype):
        return None
    if is_cutlass_available() and _is_torch_to_cutlass_available():
        if interpret_uint8_as_fp4x2 and data_type == torch.uint8:
            import cutlass

            return getattr(cutlass, "Float4E2M1FN", None)
        else:
            return _torch_to_cutlass_data_type_dict.get(data_type, None)
    return None


_cutlass_data_type_memo: dict = {}


def _convert_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2: bool = False):
    # Resolution is process-constant and this runs ~20x per grouped-GEMM wrapper call,
    # always over the same handful of dtypes. Unhashable spellings fall through uncached.
    try:
        key = (data_type, interpret_uint8_as_fp4x2)
        cached = _cutlass_data_type_memo.get(key)
    except TypeError:
        return _convert_to_cutlass_data_type_uncached(data_type, interpret_uint8_as_fp4x2)
    if cached is None:
        cached = _convert_to_cutlass_data_type_uncached(data_type, interpret_uint8_as_fp4x2)
        if cached is not None:
            _cutlass_data_type_memo[key] = cached
    return cached


def _convert_to_cutlass_data_type_uncached(data_type, interpret_uint8_as_fp4x2: bool = False):
    if is_cutlass_available():
        import cutlass

        if isinstance(data_type, type) and issubclass(data_type, cutlass.Numeric):
            if interpret_uint8_as_fp4x2 and data_type is cutlass.Uint8:
                return cutlass.Float4E2M1FN
            return data_type
        elif data_type is not None:
            cutlass_data_type = _torch_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2)
            if cutlass_data_type is None:
                cutlass_data_type = _dtype_name_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2)
            if cutlass_data_type is None:
                raise ValueError("Unsupported tensor data type.")
            return cutlass_data_type
        else:
            raise ValueError("None is not a valid tensor data type.")
    return None


def _convert_to_cutlass_data_type_or_none(data_type, interpret_uint8_as_fp4x2: bool = False):
    """Like _convert_to_cutlass_data_type but returns None for unmappable dtypes instead of raising."""
    if data_type is None or not is_cutlass_available():
        return None
    try:
        return _convert_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2)
    except ValueError:
        return None


def _cudnn_to_torch_data_type(cudnn_data_type):
    """Convert a cuDNN data type to a PyTorch data type.

    Args:
        cudnn_data_type: The cuDNN data type to convert.

    Returns:
        The PyTorch data type, or None if the conversion is not available.
    """
    if is_torch_available():
        for torch_type, cudnn_type in _torch_to_cudnn_data_type_dict.items():
            if cudnn_type == cudnn_data_type:
                return torch_type
    return None


def _library_type(input_type):
    if type(input_type) is cudnn_data_type:
        return input_type

    for cvt_fn in [
        _torch_to_cudnn_data_type,
        # Add more DL libraries to support here
    ]:
        out = cvt_fn(input_type)
        if out is not None:
            return out

    # An unmappable dtype is an unsupported GRAPH, not an internal error: say so
    # with the type callers already catch, so a routing layer can read it as
    # "the backend cannot represent this" instead of guessing from a bare
    # Exception.
    import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

    raise cudnn.cudnnGraphNotSupportedError(f"No available conversion from type {input_type} to a library type.")


def _is_torch_tensor(input_tensor) -> bool:
    if is_torch_available():
        import torch

        return isinstance(input_tensor, torch.Tensor)
    return False


def _is_jax_array(input_tensor) -> bool:
    # A jax array can only exist if jax is already imported, so probing
    # sys.modules never triggers a jax import.
    jax = sys.modules.get("jax")
    if jax is not None and isinstance(input_tensor, getattr(jax, "Array", ())):
        return True
    return type(input_tensor).__module__.startswith(("jax", "jaxlib"))


# The DLPack (code, bits) a cuDNN dtype travels as. The native variant pack
# speaks DLPack, so this is the one translation between it and the graph's
# vocabulary.
_CUDNN_TO_DLPACK_CODE_BITS = {}


def _init_dlpack_dtype_tables():
    from .frost.buffers import DTYPES

    for enum, name in _CUDNN_TO_FROST_DTYPE_NAME.items():
        code_bits = DTYPES.get(name)
        if code_bits is None:
            continue
        _CUDNN_TO_DLPACK_CODE_BITS[enum] = code_bits


# torch exports float4_e2m1fn_x2 as DLPack (kDLFloat4_e2m1fn = 17, 4 bits, 2 lanes);
# the frost dtype table has no single-fp4 row, so the pack spells it the way torch does.
_DLPACK_FP4_CODE_BITS = (17, 4)


def _dlpack_code_bits(data_type):
    """``(code, bits)`` for a cuDNN dtype, or ``(0, 0)`` when it has no DLPack
    spelling — a slot with no dtype still carries its pointer and shape."""
    if not _CUDNN_TO_DLPACK_CODE_BITS:
        _init_dlpack_dtype_tables()
    got = _CUDNN_TO_DLPACK_CODE_BITS.get(data_type)
    if got is None:
        return _DLPACK_FP4_CODE_BITS if data_type == cudnn_data_type.FP4_E2M1 else (0, 0)
    return got


def _dlpack_lanes(data_type) -> int:
    """DLPack lanes of a cuDNN dtype's storage slot: 2 for fp4 (two elements per
    slot, as torch spells ``float4_e2m1fn_x2``), 1 otherwise."""
    return 2 if data_type == cudnn_data_type.FP4_E2M1 else 1
