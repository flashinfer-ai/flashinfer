# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import torch


def is_sm120_cuda13_supported() -> bool:
    """Return whether the active test device supports the Conv3d NVFP4 path."""

    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (12, 0)
        and torch.version.cuda is not None
        and int(torch.version.cuda.split(".")[0]) >= 13
    )


SM120_CUDA13_SKIP_REASON = "SM120 NVFP4 Conv3d requires SM120 and CUDA 13+"
