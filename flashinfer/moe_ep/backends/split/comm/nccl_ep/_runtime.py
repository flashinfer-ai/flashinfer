# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider version and CUDA stream resolution for the NCCL EP adapter."""

import contextlib

from packaging.version import Version


def ep_version(provider=None):
    if provider is None:
        try:
            import nccl.ep as provider
        except ImportError:
            return None
    get_version = getattr(provider, "get_lib_version", None)
    return Version(str(get_version())) if get_version is not None else None


def torch_stream(stream, device=None):
    import torch

    # ExternalStream(0) can allocate a pooled stream in supported PyTorch builds.
    if stream == 0:
        return torch.cuda.default_stream(device=device)
    return torch.cuda.ExternalStream(stream, device=device)


def stream_context(stream, *, device=None, enabled=True):
    import torch

    return (
        torch.cuda.stream(torch_stream(stream, device))
        if enabled
        else contextlib.nullcontext()
    )


def stream_is_capturing(stream):
    import torch

    if torch.cuda.is_current_stream_capturing():
        return True
    if not torch.cuda.is_available():
        return False
    with stream_context(stream):
        return torch.cuda.is_current_stream_capturing()
