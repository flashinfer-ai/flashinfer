import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest


def _load_cuda_ipc_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "flashinfer" / "comm" / "cuda_ipc.py"
    spec = importlib.util.spec_from_file_location("_cuda_ipc_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CudaRTLibrary = _load_cuda_ipc_module().CudaRTLibrary


class _FakeFunction:
    restype = None
    argtypes = None

    def __call__(self, *args):
        return 0


class _FakeCudart:
    def __getattr__(self, name):
        if name == "cudaDeviceReset":
            raise AttributeError(name)
        return _FakeFunction()


class _FakeFullCudart:
    def __getattr__(self, name):
        return _FakeFunction()


def _clear_cudart_cache():
    CudaRTLibrary.path_to_library_cache.clear()
    CudaRTLibrary.path_to_dict_mapping.clear()


def test_auto_cudart_resolution_skips_stub_without_required_symbols():
    _clear_cudart_cache()
    maps = "\n".join(
        [
            "7f000000-7f001000 r-xp 00000000 00:00 0 /pkg/tilelang/lib/libcudart_stub.so",
            "7f010000-7f011000 r-xp 00000000 00:00 0 /usr/local/cuda/lib64/libcudart.so.13",
        ]
    )
    libraries = {
        "/pkg/tilelang/lib/libcudart_stub.so": _FakeCudart(),
        "/usr/local/cuda/lib64/libcudart.so.13": _FakeFullCudart(),
    }

    with (
        mock.patch("builtins.open", mock.mock_open(read_data=maps)),
        mock.patch("ctypes.CDLL", side_effect=lambda path: libraries[path]) as cdll,
    ):
        cudart = CudaRTLibrary()

    assert cudart.lib is libraries["/usr/local/cuda/lib64/libcudart.so.13"]
    assert [call.args[0] for call in cdll.call_args_list] == [
        "/pkg/tilelang/lib/libcudart_stub.so",
        "/usr/local/cuda/lib64/libcudart.so.13",
    ]
    assert (
        "/pkg/tilelang/lib/libcudart_stub.so" not in CudaRTLibrary.path_to_library_cache
    )
    assert (
        "/pkg/tilelang/lib/libcudart_stub.so" not in CudaRTLibrary.path_to_dict_mapping
    )


def test_explicit_cudart_path_still_requires_exported_symbols():
    _clear_cudart_cache()
    stub_path = "/pkg/tilelang/lib/libcudart_stub.so"
    with (
        mock.patch("ctypes.CDLL", return_value=_FakeCudart()),
        pytest.raises(AttributeError, match="cudaDeviceReset"),
    ):
        CudaRTLibrary(stub_path)

    assert stub_path not in CudaRTLibrary.path_to_library_cache
    assert stub_path not in CudaRTLibrary.path_to_dict_mapping
