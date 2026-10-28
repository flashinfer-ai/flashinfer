# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MX executable loading without changing the process-default architecture."""

import gc
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from flashinfer.fused_moe import QuantFormat
from flashinfer.utils import is_sm100a_supported
from tests.moe.test_frost_low_latency_mx_contracts import (
    _TensorMetadata,
    _packs,
    _runner,
)


@pytest.mark.parametrize("fmt", [QuantFormat.MXFP4, QuantFormat.MXFP8])
@pytest.mark.parametrize("top_k", [2, 6])
def test_mx_cross_target_cache_rebinds_live_inputs(monkeypatch, fmt, top_k):
    pytest.importorskip("cutlass", minversion="4.8")
    from flashinfer.fused_moe.backends.frost_low_latency import runtime

    runner = _runner(monkeypatch, fmt, top_k=top_k)
    runner.check_support()
    runner.build()
    act, pack, _ = _packs(monkeypatch, runner)
    inputs = runner.pack_inputs(act, pack)
    monkeypatch.setattr(runtime, "_COMPILED", {})
    from flashinfer.fused_moe.backends.frost_low_latency.fma import (
        runtime as fma_runtime,
    )

    monkeypatch.setattr(runtime, "from_dlpack", lambda x: x)
    monkeypatch.setattr(fma_runtime, "from_dlpack", lambda x: x)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 0))
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
        assert runner.forward(inputs) is inputs[13]
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


def _native_mx_pipeline(format_name, device_index, top_k=2):
    """Run in a child whose compiler default was set before importing CuTe."""
    from flashinfer.fused_moe.backends.frost_low_latency import runtime
    from flashinfer.fused_moe.backends.frost_low_latency.runner import (
        FrostLowLatencyRunner,
    )
    from tests.moe.test_frost_low_latency import _assert_near
    from tests.moe.test_frost_low_latency_mx_contracts_gpu import _case

    assert os.environ["CUTE_DSL_ARCH"] == "sm_90a"
    torch.cuda.set_device(device_index)
    assert torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    runtime._COMPILED.clear()
    load_events = []
    original_load = runtime._load_cross_compiled

    def record_load(entry):
        load_events.append(entry.engine is None)
        return original_load(entry)

    runtime._load_cross_compiled = record_load
    held = []
    runner = None
    for step in range(3):
        config, act, pack, expected = _case(
            QuantFormat[format_name], top_k=top_k, seed=8163 + step
        )
        act.topk_ids.add_(step).remainder_(config.routing.num_experts)
        act.topk_weights.mul_(step + 1).sub_(0.25)
        if runner is None:
            runner = FrostLowLatencyRunner(config, act.hidden_states_q.device)
            runner.check_support()
            runner.build()
        inputs = runner.pack_inputs(act, pack)
        # Holding every call's tensors prevents allocator reuse from concealing
        # stale bindings; different seeds expose stale activation/weight values.
        # Default dispatch exercises K128 for top_k=2 and
        # FFMA for top_k=6, which exceeds the four available experts.
        held.append(inputs)
        _assert_near(runner.forward(inputs), expected())
    assert load_events == [True]
    assert len({inputs[0].data_ptr() for inputs in held}) == 3
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner.forward(inputs)
    for step in range(3):
        act.topk_ids.add_(1).remainder_(config.routing.num_experts)
        act.topk_weights.fill_(0.125 * (step + 1))
        # Packed score conversion was allocated outside capture; update the
        # actual bound input as well as the FP32 reference scores.
        inputs[3].copy_(act.topk_weights)
        graph.replay()
        _assert_near(output, expected())
    assert load_events == [True]
    assert os.environ["CUTE_DSL_ARCH"] == "sm_90a"
    print(
        f"native {format_name}, top_k={top_k}: "
        f"3 calls, 3 replays, load fallbacks={load_events}"
    )


@pytest.mark.parametrize("format_name", ["MXFP4", "MXFP8"])
@pytest.mark.parametrize("top_k", [2, 6])
def test_native_mx_cross_arch_pipeline_in_fresh_process(format_name, top_k, tmp_path):
    pytest.importorskip("cutlass", minversion="4.8")
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
        or not is_sm100a_supported(torch.device("cuda", torch.cuda.current_device()))
    ):
        pytest.skip(
            "Native MX cross-architecture regression requires SM100/SM103 and CUDA 12.8+"
        )
    environment = os.environ.copy()
    environment["CUTE_DSL_ARCH"] = "sm_90a"
    # Changing the compiler target needs a private cache namespace. Inherit
    # cache validation, but never reuse or invalidate another worker's cache.
    environment.update(
        FLASHINFER_WORKSPACE_BASE=str(tmp_path),
        FLASHINFER_CACHE_DIR=str(tmp_path / ".cache" / "flashinfer"),
        CUTE_DSL_CACHE_DIR=str(tmp_path / "cute_dsl"),
        TRITON_CACHE_DIR=str(tmp_path / "triton"),
        TORCH_EXTENSIONS_DIR=str(tmp_path / "torch_extensions"),
        TVM_FFI_CACHE_DIR=str(tmp_path / "tvm_ffi"),
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tests.moe.test_frost_low_latency_mx_jit import "
            "_native_mx_pipeline; "
            "_native_mx_pipeline(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))",
            format_name,
            str(torch.cuda.current_device()),
            str(top_k),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        text=True,
        capture_output=True,
        timeout=600,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    print(child.stdout)
