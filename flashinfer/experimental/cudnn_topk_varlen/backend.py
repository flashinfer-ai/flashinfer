# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Public cuDNN prepared API adapter; cache owns metadata and code only."""

import functools
import importlib
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version
import torch


@functools.cache
def frontend_api(cc):
    """Optional feature detection, without hiding compilation/launch errors."""
    try:
        dsl_version = Version(version("nvidia-cutlass-dsl"))
    except PackageNotFoundError:
        return None
    if dsl_version < Version("4.8.0" if cc == 107 else "4.7.0"):
        return None
    try:
        cudnn = importlib.import_module("cudnn")
    except ModuleNotFoundError as exc:
        if exc.name == "cudnn":
            return None
        raise
    return getattr(cudnn, "IndexerTopKVarlen", None)


# A captured graph can outlive its last Python invocation. Keep the plan's
# compiled-module owner alive for the process; there is no graph-close hook
# through which an LRU could safely release it. Entries contain no GPU tensors.
@functools.cache
def _prepare(device_index, rows, cols, top_k, next_n, compress_ratio):
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Warm up top_k_varlen backend='cudnn' eagerly before CUDA graph capture"
            )
        major, minor = torch.cuda.get_device_capability(device)
        api = frontend_api(major * 10 + minor)
        if api is None:
            raise ImportError(
                "backend='cudnn' requires a cudnn-frontend build exposing "
                "IndexerTopKVarlen and nvidia-cutlass-dsl >= 4.7 (SM107: >= 4.8)"
            )
        from cudnn.api_base import TensorDesc

        scores = TensorDesc(
            dtype=torch.bfloat16,
            shape=(rows, cols),
            stride=(cols, 1),
            stride_order=(1, 0),
            device=device,
        )
        lengths = TensorDesc(
            dtype=torch.int32,
            shape=(rows // next_n,),
            stride=(1,),
            stride_order=(0,),
            device=device,
        )
        plan = api(scores, lengths, top_k, next_n, compress_ratio)
        plan.compile()
        return plan


def run(logits, seq_lens, top_k, next_n, compress_ratio, out_indices):
    # The FE plan validates every fresh tensor (including aliases/lazy flags)
    # and records its allocation on the current stream. No pointer, tensor,
    # stream or GPU workspace is retained by this metadata-keyed cache.
    plan = _prepare(
        logits.device.index,
        *logits.shape,
        top_k,
        next_n,
        compress_ratio,
    )
    plan.execute(logits, seq_lens, out_indices)
    return out_indices, None
