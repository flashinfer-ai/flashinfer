"""Kernel residency of the Cake MSA ICP indexer package (warmup before the first exchange).

``warmup`` must load every kernel of the device's architecture into the device's context: under CUDA lazy module
loading a publisher loaded at its first launch can stall behind a peer's merge that already spins on its rows.  The
package resolves each kernel's host stub from the library's symbol table, turns it into the context's function handle
(``cudaGetFuncBySymbol``) and loads it (``cuFuncLoad``), with ``cudaFuncGetAttributes`` on the stub as the second
path.  Both are called in the runtime the library binds to, not in a ``libcudart.so`` loaded by name (a profiler can
make that a second runtime copy).  An unresolvable stub, or a kernel neither path loads, must raise, never skip
residency.
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

PACKAGE = os.environ.get(
    "CAKE_MSA_ICP_PACKAGE", "flashinfer.experimental.cake_msa_icp_indexer"
)
cake_jit = importlib.import_module(f"{PACKAGE}.cake_jit")
cake_backend = importlib.import_module(f"{PACKAGE}.cake_backend")


def _arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    return cake_jit.arch_for_capability(torch.cuda.get_device_capability(0))


requires_arch = pytest.mark.skipif(
    _arch() is None, reason="needs an sm_100a / sm_103a / sm_107a GPU"
)


def _records(arch: str) -> list[str]:
    return sorted(
        name for name, record in cake_jit.MODULES.items() if arch in record["arches"]
    )


@requires_arch
def test_warmup_makes_every_kernel_of_the_architecture_resident() -> None:
    arch = _arch()
    names = _records(arch)
    assert names, f"no program ships for {arch}"
    cake_backend.warmup(torch.device("cuda", 0))
    device = torch.cuda.current_device()
    missing = [name for name in names if (name, device) not in cake_jit._RESIDENT]
    assert not missing


@requires_arch
def test_every_kernel_stub_resolves_from_the_library_symbol_table() -> None:
    cake_build = importlib.import_module(f"{PACKAGE}.cake_build")
    arch = _arch()
    for name in _records(arch):
        cake_jit.load(name, arch)
        library = cake_jit.library_name(name, arch)
        path = str(cake_build.library_path(library))
        if (
            os.environ.get("CAKE_MSA_ICP_EXPECT_AOT") == "1"
        ):  # a vendored build: the builder's aot/ library
            assert path == str(cake_build.prebuilt_path(library)), path
        values = cake_jit._symbol_values(
            path, (cake_jit.MODULES[name]["kernel_symbol"], "__tvm_ffi_run")
        )
        assert len(values) == 2, (name, path, values)


@requires_arch
def test_an_unresolvable_stub_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    arch = _arch()
    name = _records(arch)[0]
    cake_jit.load(name, arch)
    record = dict(
        cake_jit.MODULES[name], kernel_symbol="kernel_absent_from_the_library"
    )
    monkeypatch.setitem(cake_jit.MODULES, name, record)
    with pytest.raises(RuntimeError, match="lacks the symbol"):
        cake_jit._make_resident(name, arch, torch.cuda.current_device())


def _calls(name: str, arch: str):
    cake_build = importlib.import_module(f"{PACKAGE}.cake_build")
    cake_jit.load(name, arch)
    return cake_jit._runtime(
        str(cake_build.library_path(cake_jit.library_name(name, arch)))
    )


@requires_arch
def test_the_driver_path_loads_every_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    arch = _arch()
    for name in _records(arch):
        calls = _calls(name, arch)
        assert calls.cuFuncLoad is not None, (
            "the runtime gives no cuFuncLoad entry point (CUDA >= 12.4 expected)"
        )

        def refuse(*_):
            raise AssertionError(
                "cudaFuncGetAttributes called although the driver path loaded the kernel"
            )

        monkeypatch.setattr(calls, "cudaFuncGetAttributes", refuse)
        cake_jit._make_resident(name, arch, torch.cuda.current_device())
        monkeypatch.undo()


@requires_arch
def test_cudaFuncGetAttributes_serves_when_the_driver_path_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arch = _arch()
    name = _records(arch)[0]
    calls = _calls(name, arch)
    monkeypatch.setattr(
        calls, "cuFuncLoad", lambda *_: 400
    )  # CUDA_ERROR_INVALID_HANDLE
    cake_jit._make_resident(name, arch, torch.cuda.current_device())


@requires_arch
def test_a_kernel_neither_path_loads_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    arch = _arch()
    name = _records(arch)[0]
    calls = _calls(name, arch)
    monkeypatch.setattr(
        calls, "cuFuncLoad", lambda *_: 400
    )  # CUDA_ERROR_INVALID_HANDLE
    monkeypatch.setattr(
        calls, "cudaFuncGetAttributes", lambda *_: 400
    )  # cudaErrorInvalidResourceHandle
    with pytest.raises(
        RuntimeError,
        match="could not be made resident.*cuFuncLoad: 400.*cudaFuncGetAttributes: 400",
    ):
        cake_jit._make_resident(name, arch, torch.cuda.current_device())


@requires_arch
def test_residency_ignores_a_second_runtime_copy_in_the_process() -> None:
    """Nsight Systems' OS-runtime tracing (``nsys profile -t osrt``) makes a load of ``libcudart.so.13`` by name map
    the CUDA toolkit's runtime next to the one PyTorch loaded; that copy knows none of the library's kernels
    (``cudaGetFuncBySymbol`` -> invalid device function).  Map such a copy here: residency must not use it."""
    import ctypes

    arch = _arch()
    name = _records(arch)[0]
    calls = _calls(name, arch)
    with open("/proc/self/maps") as maps:
        mapped = {line.split()[-1] for line in maps if "libcudart.so" in line}
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    others = [
        candidate
        for candidate in (
            os.path.join(cuda_home, sub, "libcudart.so.13")
            for sub in (
                "lib64",
                "lib",
                "targets/sbsa-linux/lib",
                "targets/x86_64-linux/lib",
            )
        )
        if os.path.isfile(candidate)
        and os.path.realpath(candidate) not in {os.path.realpath(m) for m in mapped}
    ]
    if not others:
        pytest.skip(
            f"no second libcudart.so.13 under {cuda_home} besides the mapped {sorted(mapped)}"
        )
    foreign = ctypes.CDLL(others[0])
    foreign.cudaGetFuncBySymbol.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cake_build = importlib.import_module(f"{PACKAGE}.cake_build")
    path = str(cake_build.library_path(cake_jit.library_name(name, arch)))
    kernel = cake_jit.MODULES[name]["kernel_symbol"]
    values = cake_jit._symbol_values(path, (kernel, "__tvm_ffi_run"))
    entry = ctypes.cast(calls.library.__tvm_ffi_run, ctypes.c_void_p).value
    stub = ctypes.c_void_p(entry - values["__tvm_ffi_run"] + values[kernel])
    function = ctypes.c_void_p()
    assert foreign.cudaGetFuncBySymbol(ctypes.byref(function), stub) != 0, (
        "the second runtime copy knows the kernel"
    )
    foreign.cudaGetLastError()
    cake_jit._make_resident(name, arch, torch.cuda.current_device())
