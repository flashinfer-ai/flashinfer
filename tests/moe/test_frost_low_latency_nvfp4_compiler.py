# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for the loaded NVFP4 compiler admission boundary."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    FrostLowLatencyConfig,
    MoEConfig,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner


def _runner(dtype, arch, monkeypatch):
    monkeypatch.setattr("flashinfer.utils.get_compute_capability", lambda device: arch)
    config = MoEConfig(
        RoutingConfig(32, 10),
        QuantConfig(weight=QuantFormat.NVFP4, activation=dtype, output=dtype),
        ExpertConfig(512),
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    runner = FrostLowLatencyRunner(config, torch.device("cpu"))
    runner.check_support()
    runner.build()
    return runner


@pytest.mark.parametrize("dtype", [QuantFormat.BF16, QuantFormat.FP16])
@pytest.mark.parametrize("arch", [(10, 3), (12, 0)])
@pytest.mark.parametrize("compiler_major", [12, 13, 14, None])
def test_nvfp4_compiler_admission(dtype, arch, compiler_major, monkeypatch):
    cutlass = pytest.importorskip("cutlass", minversion="4.8")
    # Torch CUDA 13 must not admit the broken cu12 compiler, nor does Torch
    # CUDA 12 imply a cu12 DSL compiler. Missing/future components fail closed.
    monkeypatch.setattr(
        torch.version, "cuda", "12.9" if compiler_major == 13 else "13.0"
    )
    if compiler_major is None:
        monkeypatch.delattr(cutlass, "CUDA_VERSION")
    else:
        monkeypatch.setattr(
            cutlass, "CUDA_VERSION", SimpleNamespace(major=compiler_major)
        )
    runner = _runner(dtype, arch, monkeypatch)
    inputs = [torch.empty(8, 2048)] + [None] * 12
    tactics = runner.get_valid_tactics(inputs, None)
    assert len([v for v in tactics if v[0] == "ffma"]) == 64
    tc = [v for v in tactics if v[0] == "warp_mma"]
    if arch == (10, 3):
        assert tc == [("warp_mma", 4, 2, 256), ("warp_mma", 8, 2, 256)]
    else:
        assert tc[:3] == [
            ("warp_mma", 4, 2, 128),
            ("warp_mma", 8, 4, 128),
            ("warp_mma", 4, 2, 256),
        ]
        assert (("warp_mma", 8, 2, 256) in tc) == (compiler_major == 13)
        assert len(tc) == (4 if compiler_major == 13 else 3)
        if compiler_major != 13:

            def no_compile(*args, **kwargs):
                raise AssertionError("Rejected candidate must not compile or launch")

            monkeypatch.setattr(cutlass.cute, "compile", no_compile)
            # Also cover a persisted/explicit tactic bypassing enumeration.
            with pytest.raises(ValueError, match="loaded CuTe DSL cu13 compiler"):
                runner.forward(inputs, ("warp_mma", 8, 2, 256))


def test_nvfp4_compiler_tuning_cache_key(monkeypatch):
    cutlass = pytest.importorskip("cutlass", minversion="4.8")
    runner = _runner(QuantFormat.BF16, (12, 0), monkeypatch)
    keys = []
    for major in (12, 13, None):
        monkeypatch.setattr(cutlass, "CUDA_VERSION", SimpleNamespace(major=major))
        keys.append(runner._cache_key_extras())
    assert len(set(keys)) == 3


@pytest.mark.parametrize("dtype_name", ["BFloat16", "Float16"])
def test_nvfp4_rejects_weight_first_projection(dtype_name):
    cutlass = pytest.importorskip("cutlass", minversion="4.8")
    from cutlass import cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    from flashinfer.fused_moe.backends.frost_low_latency.tc.warp import (
        WarpGroupedProjection,
    )

    def tensor(dtype, shape):
        strides = []
        stride = 1
        for size in reversed(shape):
            strides.append(stride)
            stride *= size
        return make_fake_tensor(
            dtype, shape, tuple(reversed(strides)), assumed_align=16
        )

    dtype = getattr(cutlass, dtype_name)
    projection = WarpGroupedProjection(True, 4, 2, 128, 10, weight_first=True)
    # Exercise the actual JIT entry with metadata only; no GPU launch. A serial
    # producer cannot initialize the scale rows owned by the other three warps.
    with pytest.raises(ValueError, match="Packed NVFP4 requires cooperative gathering"):
        cute.compile(
            projection,
            tensor(cutlass.Uint8, (19, 1024, 1024)),
            tensor(dtype, (8, 2048)),
            tensor(cutlass.Int32, (8, 10)),
            tensor(dtype, (80, 512)),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            tensor(cutlass.Float32, (19, 2)),
            None,
            tensor(cutlass.Uint8, (19, 1024, 128)),
            options="--gpu-arch sm_120a --enable-tvm-ffi",
        )
