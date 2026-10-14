"""B3 — NDTensor unit tests (mocked NCCL library).

These tests don't need GPU comms: they patch ``flashinfer.moe_ep.backends.split.comm.nccl_ep.ndtensor.get_nccl_lib``
to return a fake library whose ``_funcs`` dict records the call arguments. Real
``ncclEpTensorCreate`` behavior is exercised in the on-cluster smoke tests
(``smoke_nccl_ep.py``) and the B4 mock-based Fleet tests.
"""

from __future__ import annotations

import ctypes
from unittest import mock

import pytest


@pytest.fixture
def fake_nccl_lib():
    """A fake NCCLLibrary whose ``_funcs[name]`` records (args) on call."""
    calls: dict[str, list[tuple]] = {
        "ncclEpTensorCreate": [],
        "ncclEpTensorDestroy": [],
        "ncclEpTensorGetData": [],
        "ncclEpTensorGetSizes": [],
    }

    def make_recorder(name):
        def recorder(*args):
            calls[name].append(args)
            return 0  # ncclSuccess

        return recorder

    lib = mock.Mock()
    lib._funcs = {name: make_recorder(name) for name in calls}
    lib.NCCL_CHECK = lambda rc: None
    lib._calls = calls  # convenience handle for assertions
    return lib


@pytest.fixture
def fake_nccl_ep_module():
    """Patch ``nccl_ep`` so dtype enum + ncclNDTensor_t are available."""
    import sys

    if "nccl_ep" in sys.modules:
        yield sys.modules["nccl_ep"]
        return

    fake_mod = mock.Mock()
    fake_mod.ncclNDTensor_t = ctypes.c_void_p

    class _Dtypes:
        ncclInt8 = 0
        ncclUint8 = 1
        ncclInt32 = 2
        ncclInt64 = 4
        ncclFloat16 = 6
        ncclFloat32 = 7
        ncclFloat64 = 8
        ncclBfloat16 = 9

        @classmethod
        def from_torch(cls, dtype):
            import torch

            return {
                torch.bfloat16: cls.ncclBfloat16,
                torch.float32: cls.ncclFloat32,
                torch.int64: cls.ncclInt64,
            }[dtype]

    fake_mod.ncclDataTypeEnum = _Dtypes
    sys.modules["nccl_ep"] = fake_mod
    yield fake_mod
    del sys.modules["nccl_ep"]


def test_from_torch_calls_create(fake_nccl_lib, fake_nccl_ep_module):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for tensor.is_cuda check")

    from flashinfer.moe_ep.backends.split.comm.nccl_ep import ndtensor

    with mock.patch.object(ndtensor, "get_nccl_lib", return_value=fake_nccl_lib):
        x = torch.zeros(32, 4096, dtype=torch.bfloat16, device="cuda")
        nd = ndtensor.NDTensor.from_torch(
            group=ctypes.c_void_p(0xCAFE), tensor=x, tag=1
        )

    assert nd.shape == (32, 4096)
    assert nd.dtype is torch.bfloat16
    assert nd.tag == 1
    assert not nd._owns
    # Verify the ncclEpTensorCreate call shape: (group, byref(out), ndim, dtype, tag, data, s0, s1, ..., s4)
    assert len(fake_nccl_lib._calls["ncclEpTensorCreate"]) == 1
    args = fake_nccl_lib._calls["ncclEpTensorCreate"][0]
    # args[0] = group (we passed 0xCAFE)
    assert args[0].value == 0xCAFE
    # args[2] = ndim, args[3] = dtype enum, args[4] = tag
    assert args[2] == 2  # 2D tensor
    assert args[3] == 9  # ncclBfloat16
    assert args[4] == 1  # tag
    # args[6..10] are sizes (padded to 5D)
    assert args[6:11] == (32, 4096, 0, 0, 0)


def test_from_torch_rejects_noncontiguous(fake_nccl_lib, fake_nccl_ep_module):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep.backends.split.comm.nccl_ep import ndtensor

    with mock.patch.object(ndtensor, "get_nccl_lib", return_value=fake_nccl_lib):
        x = torch.zeros(64, 4096, device="cuda").T  # non-contiguous transpose
        with pytest.raises(ValueError, match="contiguous"):
            ndtensor.NDTensor.from_torch(ctypes.c_void_p(0), x, tag=1)


def test_from_torch_rejects_cpu(fake_nccl_lib, fake_nccl_ep_module):
    import torch

    from flashinfer.moe_ep.backends.split.comm.nccl_ep import ndtensor

    with mock.patch.object(ndtensor, "get_nccl_lib", return_value=fake_nccl_lib):
        x = torch.zeros(32, 4096)  # CPU
        with pytest.raises(ValueError, match="CUDA"):
            ndtensor.NDTensor.from_torch(ctypes.c_void_p(0), x, tag=1)


def test_allocate_owns_handle(fake_nccl_lib, fake_nccl_ep_module):
    import torch

    from flashinfer.moe_ep.backends.split.comm.nccl_ep import ndtensor

    with mock.patch.object(ndtensor, "get_nccl_lib", return_value=fake_nccl_lib):
        nd = ndtensor.NDTensor.allocate(
            group=ctypes.c_void_p(0xBEEF),
            dtype=torch.bfloat16,
            shape=(64, 4096),
            tag=1,
        )
    assert nd._owns is True
    # Allocate path passes data=nullptr (args[5]).
    args = fake_nccl_lib._calls["ncclEpTensorCreate"][0]
    assert args[5].value in (0, None)  # ctypes.c_void_p(0)


def test_destroy_on_del(fake_nccl_lib, fake_nccl_ep_module):
    import torch

    from flashinfer.moe_ep.backends.split.comm.nccl_ep import ndtensor

    with mock.patch.object(ndtensor, "get_nccl_lib", return_value=fake_nccl_lib):
        nd = ndtensor.NDTensor.allocate(
            group=ctypes.c_void_p(0xBEEF),
            dtype=torch.bfloat16,
            shape=(64, 4096),
            tag=1,
        )
        del nd
        # ncclEpTensorDestroy gets called via __del__ → owns=True path.
    assert len(fake_nccl_lib._calls["ncclEpTensorDestroy"]) == 1
