# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""Launch CuTe-DSL kernels on torch tensors. A kernel compiles once per kind of
argument (dtype, rank, alignment, and whether it needs 64-bit offsets), for any
shapes and row strides, and is cached on disk."""

import hashlib
import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel

CUTLASS_TYPES = {
    torch.int16: cutlass.Int16,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
    torch.uint8: cutlass.Uint8,
    torch.float32: cutlass.Float32,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}


def argument_kind(arg: object) -> object:
    """The part of an argument a kernel compiles for: a tensor's dtype, rank,
    alignment in bytes (its element size for one dimension, else the largest
    power of two up to 16 dividing a row) and whether it spans 2**31 elements or
    more, or else the type (int or None). TVM-FFI rejects less aligned tensors."""
    if not isinstance(arg, torch.Tensor):
        return type(arg)
    size = arg.element_size()
    align = math.gcd(arg.shape[-1] * size, 16) if arg.dim() > 1 else size
    return arg.dtype, arg.dim(), align, arg.shape[0] * arg.stride(0) >= 2**31


def fake_argument(kind: object) -> object:
    """A symbolic argument of an `argument_kind`. Tensors take any shape and row
    strides, with a unit last stride. A tensor of 2**31 elements or more, such
    as a large cache, gets 64-bit row strides so its offsets stay exact."""
    if not isinstance(kind, tuple):
        return cutlass.Int32(0) if kind is int else None
    dtype, rank, align, wide = kind
    divisibility = max(align // dtype.itemsize, 1)
    stride = cute.sym_int64 if wide else cute.sym_int
    return cute.runtime.make_fake_tensor(
        CUTLASS_TYPES[dtype],
        tuple(cute.sym_int() for _ in range(rank)),
        tuple(stride(divisibility=divisibility) for _ in range(rank - 1)) + (1,),
        assumed_align=max(align, dtype.itemsize),
    )


@cache
def compiled(
    function,
    module: str,
    name: str,
    constants: tuple,
    device,
    kinds: tuple,
    key_files: tuple,
):
    """Compile `function`, a JIT function of a stream, `constants` and arguments
    of `kinds`, or load it from the disk cache, where it is `name` in `module`.
    A change to `key_files` or this file invalidates the cache."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"Warm up {module} kernels before CUDA graph capture")
    key = repr(([getattr(c, "__name__", c) for c in constants], kinds)).encode()
    with torch.cuda.device(device):
        return build_and_load_cute_dsl_kernel(
            module,
            f"{name}_{hashlib.sha256(key).hexdigest()[:16]}",
            lambda: cute.compile(
                function,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                *constants,
                *map(fake_argument, kinds),
                options="--enable-tvm-ffi",
            ),
            extra_key_files=(__file__, *key_files),
        )


@cute.jit
def launcher(
    stream: cuda.CUstream,
    kernel: cutlass.Constexpr,
    threads: cutlass.Constexpr[int],
    constants: cutlass.Constexpr[tuple],
    blocks: cutlass.Int32,
    tiles: cutlass.Int32,
    *args,
):
    kernel(*args, *constants).launch(
        grid=(blocks, tiles, 1), block=(threads, 1, 1), stream=stream
    )


def launch(
    kernel,
    grid: tuple,
    threads: int,
    args: tuple,
    constants: tuple,
    *,
    module: str,
    key_files: tuple = (),
) -> None:
    """Run `kernel` on `args` (tensors, ints or Nones) and `constants`, over a
    grid of CTAs of `threads` threads. It compiles once per kind of argument,
    and is cached on disk in `module` until `key_files` change."""
    kinds = (int, int, *map(argument_kind, args))
    constants = (kernel, threads, constants)
    device = args[0].device
    function = compiled(
        launcher, module, kernel.__name__, constants, device, kinds, key_files
    )
    function(*grid, *args)
