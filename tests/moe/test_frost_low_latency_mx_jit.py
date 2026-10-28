# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MX executable loading without changing the process-default architecture."""

import gc
import os
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from flashinfer.fused_moe import QuantFormat
from tests.moe.test_frost_low_latency_mx_contracts import (
    _TensorMetadata,
    _packs,
    _runner,
)


@pytest.mark.parametrize("fmt", [QuantFormat.MXFP4, QuantFormat.MXFP8])
def test_mx_cross_target_cache_rebinds_live_inputs(monkeypatch, fmt):
    pytest.importorskip("cutlass", minversion="4.8")
    from flashinfer.fused_moe.backends.frost_low_latency import runtime

    runner = _runner(monkeypatch, fmt)
    runner.check_support()
    runner.build()
    act, pack, _ = _packs(monkeypatch, runner)
    inputs = runner.pack_inputs(act, pack)
    monkeypatch.setattr(runtime, "_COMPILED", {})
    monkeypatch.setattr(runtime, "from_dlpack", lambda x: x)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    capturing = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    compiled, exports, loads, calls = [], [], [], []
    original_arch = os.environ.get("CUTE_DSL_ARCH")

    class CrossCompiled:
        engine = None

        def export_to_c(self, object_file_path, function_name):
            Path(object_file_path).write_bytes(b"object fixture")
            exports.append((object_file_path, function_name))

        def __call__(self, *args):
            raise RuntimeError("TVM FFI function is not initialized")

    def compile_stub(*args, **kwargs):
        compiled.append(kwargs["options"])
        return CrossCompiled()

    def load_stub(path, *, enable_tvm_ffi):
        assert enable_tvm_ffi
        assert Path(path).read_bytes() == b"object fixture"
        loads.append(path)
        return {exports[-1][1]: lambda *live: calls.append(live)}

    monkeypatch.setattr(runtime.cute, "compile", compile_stub)
    monkeypatch.setattr(runtime.cute.runtime, "load_module", load_stub)
    monkeypatch.setattr(runtime.cute.runtime, "make_fake_stream", lambda **_: None)
    for index in range(3):
        capturing = index > 0
        inputs = [replace(v) if isinstance(v, _TensorMetadata) else v for v in inputs]
        assert runtime.execute(inputs, runner.config, 128) is inputs[13]
        assert calls[-1][0] is inputs[0]
    assert len(compiled) == len(exports) == len(loads) == 1
    assert len(calls) == 3
    assert not Path(exports[0][0]).exists()
    assert os.environ.get("CUTE_DSL_ARCH") == original_arch


def test_cross_target_host_object_survives_temporary_storage(monkeypatch, tmp_path):
    cutlass = pytest.importorskip("cutlass", minversion="4.8")
    from cutlass import cute
    from flashinfer.fused_moe.backends.frost_low_latency import runtime

    @cute.jit
    def add(output: cute.Tensor, x: cutlass.Int32, offset: cutlass.Constexpr):
        output[0] = x + offset

    monkeypatch.setattr(
        runtime, "TemporaryDirectory", partial(TemporaryDirectory, dir=tmp_path)
    )
    original_arch = os.environ.get("CUTE_DSL_ARCH")
    functions = []
    for offset in (3, -5):
        # Force the cross-compile state without needing a GPU or mutating the DSL singleton.
        entry = cute.compile(
            add,
            cute.runtime.make_fake_tensor(
                cutlass.Int32, (1,), (1,), memspace=cutlass.AddressSpace.generic
            ),
            cutlass.Int32(0),
            offset,
            no_jit_engine=True,
            options="--gpu-arch sm_100a --enable-tvm-ffi",
        )
        assert entry.engine is None
        functions.append(runtime._load_cross_compiled(entry))
        del entry
    gc.collect()
    assert not list(tmp_path.iterdir())
    for function, offset in zip(functions, (3, -5), strict=True):
        for value in (0, 7, -11):
            output = torch.full((1,), -99, dtype=torch.int32)
            function(output, value)
            assert output.item() == value + offset
    direct = cute.compile(
        add,
        cute.runtime.make_fake_tensor(
            cutlass.Int32, (1,), (1,), memspace=cutlass.AddressSpace.generic
        ),
        cutlass.Int32(0),
        9,
        options="--gpu-arch sm_100a --enable-tvm-ffi",
    )
    assert direct.engine is not None
    assert runtime._load_cross_compiled(direct) is direct
    assert os.environ.get("CUTE_DSL_ARCH") == original_arch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_cross_target_device_object_rebinds_and_replays():
    cutlass = pytest.importorskip("cutlass", minversion="4.8")
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.fused_moe.backends.frost_low_latency import runtime

    @cute.kernel
    def kernel(x: cute.Tensor, y: cute.Tensor, delta: cutlass.Int32):
        index, _, _ = cute.arch.thread_idx()
        y[index] = x[index] + delta

    @cute.jit
    def launch(x, y, delta, stream):
        kernel(x, y, delta).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    target = f"sm_{major}{minor}{'a' if major >= 9 else ''}"
    x = torch.arange(128, dtype=torch.int32, device=device)
    y = torch.empty_like(x)
    entry = cute.compile(
        launch,
        from_dlpack(x),
        from_dlpack(y),
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        no_jit_engine=True,
        options=f"--gpu-arch {target} --enable-tvm-ffi",
    )
    assert entry.engine is None
    loaded = runtime._load_cross_compiled(entry)
    del entry
    gc.collect()
    for seed in (1, 7, -9):
        x = torch.arange(128, dtype=torch.int32, device=device) + seed
        y = torch.empty_like(x)
        loaded(x, y, 3)
        torch.testing.assert_close(y, x + 3, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        loaded(x, y, -4)
    for value in (0, 11, -7, 23):
        x.fill_(value)
        graph.replay()
        torch.testing.assert_close(y, x - 4, rtol=0, atol=0)
