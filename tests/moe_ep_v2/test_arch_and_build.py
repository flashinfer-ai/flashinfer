"""Architecture and native-build probe tests (mocked, no comm libs)."""

from __future__ import annotations

from unittest import mock

import pytest


def test_validate_arch_skips_when_cuda_unavailable():
    from flashinfer.moe_ep_v2.core.validation.common import validate_arch_for_backend

    with mock.patch("torch.cuda.is_available", return_value=False):
        validate_arch_for_backend("nccl_ep")


@pytest.mark.parametrize("backend", ["nccl_ep", "nixl_ep"])
def test_validate_arch_rejects_pre_hopper(backend: str):
    from flashinfer.moe_ep_v2 import MoEEpArchError
    from flashinfer.moe_ep_v2.core.validation.common import validate_arch_for_backend

    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch("torch.cuda.get_device_capability", return_value=(8, 6)),
    ):
        with pytest.raises(MoEEpArchError, match="sm_90"):
            validate_arch_for_backend(backend)


@pytest.mark.parametrize("backend", ["nccl_ep", "nixl_ep"])
def test_validate_arch_accepts_hopper(backend: str):
    from flashinfer.moe_ep_v2.core.validation.common import validate_arch_for_backend

    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
    ):
        validate_arch_for_backend(backend)


def test_validate_mega_arch_rejects_pre_blackwell():
    from flashinfer.moe_ep_v2 import MoEEpArchError
    from flashinfer.moe_ep_v2.core.validation.common import validate_mega_arch

    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
    ):
        with pytest.raises(MoEEpArchError, match="sm_100"):
            validate_mega_arch()


def test_validate_mega_arch_accepts_blackwell():
    from flashinfer.moe_ep_v2.core.validation.common import validate_mega_arch

    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
    ):
        validate_mega_arch()


def test_require_built_unknown_backend_raises_value_error():
    import flashinfer.moe_ep_v2 as moe_ep_v2

    with pytest.raises(ValueError, match="unknown moe_ep_v2 backend"):
        moe_ep_v2._require_built("not_a_backend")


@pytest.mark.parametrize("backend", ["nccl_ep", "nixl_ep"])
def test_require_built_raises_when_libs_missing(backend: str):
    import flashinfer.moe_ep_v2 as moe_ep_v2

    from flashinfer.moe_ep_v2 import MoEEpNotBuiltError

    probe_name = "_probe_nccl_ep" if backend == "nccl_ep" else "_probe_nixl_ep"
    with mock.patch.object(moe_ep_v2, probe_name, return_value=False):
        with pytest.raises(MoEEpNotBuiltError, match=backend):
            moe_ep_v2._require_built(backend)


def test_available_backends_reflects_probes():
    import flashinfer.moe_ep_v2 as moe_ep_v2

    with (
        mock.patch.object(moe_ep_v2, "_probe_nccl_ep", return_value=True),
        mock.patch.object(moe_ep_v2, "_probe_nixl_ep", return_value=False),
    ):
        assert moe_ep_v2.available_backends() == ["nccl_ep"]

    with (
        mock.patch.object(moe_ep_v2, "_probe_nccl_ep", return_value=False),
        mock.patch.object(moe_ep_v2, "_probe_nixl_ep", return_value=True),
    ):
        assert moe_ep_v2.available_backends() == ["nixl_ep"]

    with (
        mock.patch.object(moe_ep_v2, "_probe_nccl_ep", return_value=True),
        mock.patch.object(moe_ep_v2, "_probe_nixl_ep", return_value=True),
    ):
        assert moe_ep_v2.available_backends() == ["nccl_ep", "nixl_ep"]


def test_have_nccl_ep_and_have_nixl_ep_delegate_to_probes():
    import flashinfer.moe_ep_v2 as moe_ep_v2

    with mock.patch.object(moe_ep_v2, "_probe_nccl_ep", return_value=True):
        assert moe_ep_v2.have_nccl_ep() is True
    with mock.patch.object(moe_ep_v2, "_probe_nccl_ep", return_value=False):
        assert moe_ep_v2.have_nccl_ep() is False

    with mock.patch.object(moe_ep_v2, "_probe_nixl_ep", return_value=True):
        assert moe_ep_v2.have_nixl_ep() is True
    with mock.patch.object(moe_ep_v2, "_probe_nixl_ep", return_value=False):
        assert moe_ep_v2.have_nixl_ep() is False
