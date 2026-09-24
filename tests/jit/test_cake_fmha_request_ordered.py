"""CPU tests for the request-ordered Cake FMHA NVRTC option handling."""

import dataclasses
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from flashinfer.jit import cake_fmha_request_ordered as cake_jit


@pytest.fixture
def module_spec(tmp_path):
    source = tmp_path / "kernel.cu"
    source.write_text('extern "C" __global__ void kernel() {}')
    return cake_jit.CakeFmhaRequestOrderedModuleSpec(
        name="test",
        closure_sha256="0" * 64,
        device_path=source,
        binding_path=tmp_path / "binding.cpp",
        module_ident="test",
        kernel_symbol="kernel",
        ffi_entry="run",
        compile_options=("--use_fast_math",),
        tma_workspace_bytes=384,
    )


@pytest.fixture
def nvrtc(monkeypatch):
    api = SimpleNamespace(
        nvrtcCreateProgram=Mock(return_value=(0, "program")),
        nvrtcCompileProgram=Mock(side_effect=RuntimeError("NVRTC boundary reached")),
        nvrtcDestroyProgram=Mock(return_value=(0,)),
    )
    bindings = ModuleType("cuda.bindings")
    bindings.nvrtc = api
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    return api


@pytest.mark.parametrize("variable", ["CUDA_HOME", "CUDA_PATH"])
@pytest.mark.parametrize("installation", ["cuda", "o1/cuda", "O1/cuda"])
def test_compile_accepts_cuda_include_paths(
    monkeypatch, tmp_path, module_spec, nvrtc, variable, installation
):
    cuda_home = tmp_path / installation
    include = cuda_home / "include"
    (include / "cccl" / "cuda" / "std").mkdir(parents=True)
    (include / "cuda_runtime.h").touch()
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv(variable, str(cuda_home))
    monkeypatch.setattr(
        cake_jit, "_cuda_include_dirs", cake_jit._cuda_include_dirs.__wrapped__
    )

    with pytest.raises(RuntimeError, match="NVRTC boundary reached"):
        cake_jit._compile_cubin(module_spec)

    options = nvrtc.nvrtcCompileProgram.call_args.args[2]
    assert f"-I{include.resolve()}".encode() in options
    assert f"-I{include.resolve() / 'cccl'}".encode() in options
    assert b"--use_fast_math" in options
    nvrtc.nvrtcDestroyProgram.assert_called_once_with("program")


@pytest.mark.parametrize("option", ["-O1", "-o1"])
def test_compile_rejects_explicit_o1_option(monkeypatch, module_spec, nvrtc, option):
    monkeypatch.setattr(cake_jit, "_cuda_include_dirs", lambda: ())
    module_spec = dataclasses.replace(module_spec, compile_options=(option,))

    with pytest.raises(RuntimeError, match="forbidden O1 option"):
        cake_jit._compile_cubin(module_spec)

    nvrtc.nvrtcCreateProgram.assert_not_called()
