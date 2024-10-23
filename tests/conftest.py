import os
import types

import flashinfer
import pytest
import torch
from torch.torch_version import TorchVersion
from torch.torch_version import __version__ as torch_version

TORCH_COMPILE_FNS = [
    flashinfer.norm.rmsnorm,
    flashinfer.norm.fused_add_rmsnorm,
    flashinfer.norm.gemma_rmsnorm,
    flashinfer.norm.gemma_fused_add_rmsnorm,
]


def _monkeypatch_add_torch_compile(func):
    """
    Replace the given function with its torch.compile version.
    """

    from torch._library.custom_ops import CustomOpDef

    if type(func) is types.FunctionType:
        fn = func
    elif isinstance(func, CustomOpDef):
        fn = func._init_fn
    else:
        raise ValueError(f"Unsupported fn type {type(func)}")

    components = fn.__module__.split(".")
    assert components[0] == "flashinfer"
    module = flashinfer
    for component in components[1:]:
        module = getattr(module, component)

    setattr(
        module,
        fn.__name__,
        torch.compile(
            func,
            fullgraph=True,
            backend="inductor",
            mode="max-autotune-no-cudagraphs",
        ),
    )
    print("Applied torch.compile to", f"{fn.__module__}.{fn.__name__}")


def pytest_configure(config):
    if os.environ.get("FLASHINFER_TEST_TORCH_COMPILE", "0") == "1":
        if torch_version < TorchVersion("2.4"):
            pytest.skip("torch.compile requires torch >= 2.4")
        for fn in TORCH_COMPILE_FNS:
            _monkeypatch_add_torch_compile(fn)
