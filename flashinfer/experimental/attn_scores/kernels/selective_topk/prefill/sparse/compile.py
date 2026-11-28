# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
import torch
import cutlass
import cutlass.cute as cute


def compile_call(fn, args):
    types = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
    }
    tensors = [
        cute.runtime.make_fake_compact_tensor(
            types[t.dtype],
            tuple(cute.sym_int() for _ in t.shape),
            stride_order=tuple(reversed(range(t.ndim))),
            assumed_align=16,
        )
        for t in args
    ]
    return cute.compile(
        fn,
        *tensors,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
