# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent frozen scalar oracle for the packed software FP8 SR path."""

from pathlib import Path
import hashlib
import os
import numpy as np
import torch
from torch.utils.cpp_extension import load
from flashinfer.jit import env
from .corpus import make


def verify_software_sr():
    here = Path(__file__).resolve().parent
    include = Path(env.FLASHINFER_INCLUDE_DIR)
    header = include / "flashinfer/mamba/conversion.cuh"
    before = hashlib.sha256(header.read_bytes()).hexdigest()
    major, minor = torch.cuda.get_device_capability()
    saved_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    try:
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        module = load(
            name=f"flashinfer_test_fp8_software_sr_{major}{minor}_" + before[:16],
            sources=[
                str(here / name)
                for name in ("packed_sr_probe.cu", "packed_sr_probe_binding.cpp")
            ],
            extra_include_paths=[str(include)],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    finally:
        if saved_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = saved_arch
    arrays, coverage = make()
    inputs = [torch.from_numpy(value).to("cuda") for value in arrays]
    rng = torch.cuda.get_rng_state().clone()
    reference, candidate = module.probe(*inputs)
    torch.cuda.synchronize()
    reference, candidate = (value.cpu().numpy() for value in (reference, candidate))
    assert np.array_equal(reference, candidate), (
        "packed software SR differs from original scalar"
    )
    one, both = coverage["one_way_boundary_pairs"], coverage["both_orderings_pairs"]
    for values in (reference, candidate):
        swapped = ((values[:one] & np.uint32(0x00FF00FF)) << 8) | (
            (values[:one] & np.uint32(0xFF00FF00)) >> 8
        )
        assert np.array_equal(values[one:both], swapped)
    assert all(
        np.array_equal(value.cpu().numpy(), original)
        for value, original in zip(inputs, arrays, strict=True)
    )
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert hashlib.sha256(header.read_bytes()).hexdigest() == before
    return coverage
