# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE helper JIT targets must include the architectures its adapters use."""

from pathlib import Path

import pytest

from flashinfer.compilation_context import CompilationContext
from flashinfer.jit import cubin_loader, moe_utils


@pytest.mark.parametrize(
    "target,expected",
    [
        ((9, "0a"), "compute_90a,code=sm_90a"),
        ((10, "0a"), "compute_100a,code=sm_100a"),
        ((10, "7a"), "compute_100f,code=sm_100f"),
        ((12, "0a"), "compute_120a,code=sm_120a"),
    ],
)
def test_moe_utils_keeps_supported_arch_target(monkeypatch, target, expected):
    # Exercise the real compilation-context filter and the real JIT factory.
    # Artifact download and compilation are independent of this CPU contract.
    context = CompilationContext.__new__(CompilationContext)
    context.TARGET_CUDA_ARCHS = {target}
    monkeypatch.setattr(moe_utils, "current_compilation_context", context)
    monkeypatch.setattr(
        "flashinfer.compilation_context.cutlass_supports_sm107", lambda: False
    )
    monkeypatch.setattr(cubin_loader, "get_artifact", lambda *args: Path("header.h"))
    monkeypatch.setattr(cubin_loader, "get_meta_hash", lambda *args: "unused")
    monkeypatch.setattr(cubin_loader, "ensure_symlink", lambda *args: None)
    monkeypatch.setattr(cubin_loader, "verify_symlinked_headers", lambda *args: None)
    monkeypatch.setattr(moe_utils, "gen_jit_spec", lambda *args, **kwargs: kwargs)

    spec = moe_utils.gen_moe_utils_module()
    assert "-gencode=arch=" + expected in spec["extra_cuda_cflags"]
