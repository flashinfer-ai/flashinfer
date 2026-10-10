"""Explicit SGLang integration for short-K SM120 block-FP8 projections.

Only the GEMM scheduling policy changes. The existing weight, activation
quantization, scale layout, and output dtype are retained. This local research
module is not imported by FlashInfer's public API or automatic dispatch.
"""

from functools import lru_cache
from pathlib import Path

import torch

from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.gemm.fp8_blockwise_gemm import _fp8_blockwise_cuda_flags
from sglang.srt.layers.quantization import fp8_utils


@lru_cache(None)
def load_module():
    return load_jit(
        "sm120_projection_policy",
        external_cuda_files=[str(Path(__file__).with_name("projection_policy.cuh"))],
        cuda_wrappers=[("projection_fp8_mm", "projection_fp8_mm")],
        extra_dependencies=["cutlass"],
        extra_cuda_cflags=_fp8_blockwise_cuda_flags(),
    )


def policy_mm(a, b, sa, sb, out_dtype):
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype)
    load_module().projection_fp8_mm(out, a, b, sa, sb)
    return out


_installed = False


def install():
    """Select ordinary GEMM scheduling for M >= 1024 and K <= 2048.

    Call once at startup on the intended SM120 device. Small decode batches
    and long-K projections retain the existing SGLang implementation. QKV-A
    padding is deliberately absent: it showed no additional service gain.
    """
    global _installed
    if _installed:
        return
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This projection experiment is validated on SM120 only")
    load_module()
    old_mm = fp8_utils.fp8_blockwise_scaled_mm

    def mm(a, b, sa, sb, out_dtype):
        if a.shape[0] >= 1024 and a.shape[1] <= 2048:
            return policy_mm(a, b, sa, sb, out_dtype)
        return old_mm(a, b, sa, sb, out_dtype)

    fp8_utils.fp8_blockwise_scaled_mm = mm
    _installed = True
    print("SM120_PROJECTION_POLICY_INSTALLED", flush=True)


def create_hook(config):
    from materialized_fp8_hook import create_hook as prefill_hook

    hook = prefill_hook(config)
    install()
    return hook
