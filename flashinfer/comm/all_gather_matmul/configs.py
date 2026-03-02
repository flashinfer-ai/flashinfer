# Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared configuration for all-gather matmul kernels."""

import os
import torch

from torch.torch_version import TorchVersion


class Configs:
    TRANSFER: str
    SIGNAL_DTYPE: torch.dtype
    initialized: bool = False

    @classmethod
    def initialize(cls):
        if cls.initialized:
            return

        # Torch version >= 2.11 is required for implicit MemPool in `symm_mem.empty`
        if TorchVersion(torch.__version__) < TorchVersion("2.11"):
            raise ValueError("all_gather_matmul requires Torch version >= 2.11")

        transfer = os.getenv("TRANSFER", "CE")
        if transfer == "CE":  # Copy Engine
            cls.SIGNAL_DTYPE = torch.uint32
            cls.TRANSFER = "CE"
        elif transfer == "NVSHMEM":  # NVSHMEM
            cls.SIGNAL_DTYPE = torch.uint64
            cls.TRANSFER = "NVSHMEM"
        else:
            raise ValueError(f"Invalid transfer method: {transfer}")
        cls.initialized = True
