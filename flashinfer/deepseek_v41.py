# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Experimental DeepSeek V4.1 attention output projection."""

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def deepseek_v41_woa_plan(weight, scales, *, backend="frost"):
    """Prepare the first grouped attention output projection for single-token decode.

    Contiguous CUDA E4M3 weight [8192,4096] contains eight [1024,4096]
    matrices. E8M0 scales [256,128] describe 32-by-32 matrix blocks and may
    use uint8 storage or torch.float8_e8m0fnu. Codes 1..254 are supported.
    This is the checkpoint's two-dimensional block layout, not a rowwise
    MXFP8 GEMM layout. Weight values are decoded in FP32, rounded to BF16,
    and multiplied by BF16 inputs with FP32 accumulation and BF16 output.

    The explicit Frost backend currently supports SM100. Create the opaque
    plan outside CUDA Graph capture; creation validates scale codes and
    synchronizes. The plan retains the original allocations without copying
    weights. Keep their values, shapes and storage immutable while using it.
    Warm up deepseek_v41_woa once before capture. This inference operation
    provides no autograd implementation and performs no input quantization.
    """
    from .experimental.deepseek_v41.woa import make_woa_plan

    return make_woa_plan(weight, scales, backend=backend)


@flashinfer_experimental_api
def deepseek_v41_woa(x, plan, *, out=None):
    """Apply a prepared WOA projection to contiguous BF16 [1,8,4096] input.

    Returns contiguous BF16 [1,8,1024] on the plan's CUDA device. Use a plan
    from deepseek_v41_woa_plan and finite inputs/decoded weights. Supply out
    to reuse output storage; it must not overlap input or weight/scale
    storage. Without out, each call allocates its own output. The plan owns
    no reusable output or activation workspace. Calls use the current CUDA
    stream and support graph replay with changed input values after warmup.
    This is the WOA stage only; inverse RoPE and WOB are separate operations.
    """
    from .experimental.deepseek_v41.woa import woa

    return woa(x, plan, out=out)
