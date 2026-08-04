"""Architecture and native-build probe tests (mocked, no comm libs)."""

from __future__ import annotations

import contextlib
from unittest import mock

import pytest


def test_validate_arch_skips_when_cuda_unavailable():
    from flashinfer.moe_ep.core.validation.common import validate_arch_for_backend

    with (
        mock.patch("torch.version.cuda", "13.0"),
        mock.patch("torch.cuda.is_available", return_value=False),
    ):
        validate_arch_for_backend("nccl_ep")


@pytest.mark.parametrize("backend", ["nccl_ep", "nixl_ep"])
def test_validate_arch_rejects_pre_hopper(backend: str):
    from flashinfer.moe_ep import MoEEpArchError
    from flashinfer.moe_ep.core.validation.common import validate_arch_for_backend

    with (
        mock.patch("torch.version.cuda", "13.0"),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch("torch.cuda.get_device_capability", return_value=(8, 6)),
        pytest.raises(MoEEpArchError, match="sm_90"),
    ):
        validate_arch_for_backend(backend)


@pytest.mark.parametrize("backend", ["nccl_ep", "nixl_ep"])
def test_validate_arch_accepts_hopper(backend: str):
    from flashinfer.moe_ep.core.validation.common import validate_arch_for_backend

    with (
        mock.patch("torch.version.cuda", "13.0"),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
    ):
        validate_arch_for_backend(backend)


def test_validate_mega_arch_rejects_pre_blackwell():
    from flashinfer.moe_ep import MoEEpArchError
    from flashinfer.moe_ep.core.validation.common import validate_mega_arch

    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        pytest.raises(MoEEpArchError, match="sm_100"),
    ):
        validate_mega_arch()


def test_validate_mega_arch_accepts_blackwell():
    from flashinfer.moe_ep.core.validation.common import validate_mega_arch

    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
    ):
        validate_mega_arch()


def test_require_built_unknown_backend_raises_value_error():
    import flashinfer.moe_ep as moe_ep

    with pytest.raises(ValueError, match="unknown moe_ep backend"):
        moe_ep._require_built("not_a_backend")


@pytest.mark.parametrize("backend", ["nccl_ep", "nixl_ep"])
def test_require_built_raises_when_libs_missing(backend: str):
    import flashinfer.moe_ep as moe_ep

    from flashinfer.moe_ep import MoEEpNotBuiltError

    probe_name = "_probe_nccl_ep" if backend == "nccl_ep" else "_probe_nixl_ep"
    with (
        mock.patch.object(moe_ep, probe_name, return_value=False),
        pytest.raises(MoEEpNotBuiltError, match=backend),
    ):
        moe_ep._require_built(backend)


def test_available_backends_reflects_probes():
    import flashinfer.moe_ep as moe_ep

    with (
        mock.patch.object(moe_ep, "_probe_nccl_ep", return_value=True),
        mock.patch.object(moe_ep, "_probe_nixl_ep", return_value=False),
    ):
        assert moe_ep.available_backends() == ["nccl_ep"]

    with (
        mock.patch.object(moe_ep, "_probe_nccl_ep", return_value=False),
        mock.patch.object(moe_ep, "_probe_nixl_ep", return_value=True),
    ):
        assert moe_ep.available_backends() == ["nixl_ep"]

    with (
        mock.patch.object(moe_ep, "_probe_nccl_ep", return_value=True),
        mock.patch.object(moe_ep, "_probe_nixl_ep", return_value=True),
    ):
        assert moe_ep.available_backends() == ["nccl_ep", "nixl_ep"]


def test_have_nccl_ep_and_have_nixl_ep_delegate_to_probes():
    import flashinfer.moe_ep as moe_ep

    with mock.patch.object(moe_ep, "_probe_nccl_ep", return_value=True):
        assert moe_ep.have_nccl_ep() is True
    with mock.patch.object(moe_ep, "_probe_nccl_ep", return_value=False):
        assert moe_ep.have_nccl_ep() is False

    with mock.patch.object(moe_ep, "_probe_nixl_ep", return_value=True):
        assert moe_ep.have_nixl_ep() is True
    with mock.patch.object(moe_ep, "_probe_nixl_ep", return_value=False):
        assert moe_ep.have_nixl_ep() is False


# --------------------------------------------- CUDA-13 gate (EP runtime wheels)


@pytest.mark.parametrize("backend", ["nccl_ep", "nixl_ep"])
def test_validate_arch_rejects_cuda12_torch(backend: str):
    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.core.validation.common import validate_arch_for_backend

    with (
        mock.patch("torch.version.cuda", "12.8"),
        pytest.raises(MoEEpConfigError, match="CUDA 13"),
    ):
        validate_arch_for_backend(backend)


def test_validate_arch_skips_cuda_gate_when_version_missing():
    """A None torch.version.cuda (CPU build / exotic torch) must not raise the
    CUDA gate — the backend probes catch missing libs later."""
    from flashinfer.moe_ep.core.validation.common import validate_arch_for_backend

    with (
        mock.patch("torch.version.cuda", None),
        mock.patch("torch.cuda.is_available", return_value=False),
    ):
        validate_arch_for_backend("nccl_ep")


# --------------------------------------- Blackwell NCCL >= 2.30.7 floor (nccl_ep)


@contextlib.contextmanager
def _cuda13_env(capability):
    with (
        mock.patch("torch.version.cuda", "13.0"),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch("torch.cuda.get_device_capability", return_value=capability),
    ):
        yield


def _fake_nccl_version(ver):
    return mock.patch(
        "flashinfer.moe_ep.core.validation.common._installed_nccl_version",
        return_value=ver,
    )


def test_validate_arch_rejects_old_nccl_on_blackwell():
    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.core.validation.common import validate_arch_for_backend

    with (
        _cuda13_env((10, 0)),
        _fake_nccl_version((2, 29, 7)),
        pytest.raises(MoEEpConfigError, match="2.30.7"),
    ):
        validate_arch_for_backend("nccl_ep")


@pytest.mark.parametrize("nccl_ver", [(2, 30, 7), (2, 31, 0), None])
def test_validate_arch_accepts_new_or_unknown_nccl_on_blackwell(nccl_ver):
    """>= floor passes; an undeterminable version must NOT block (None case)."""
    from flashinfer.moe_ep.core.validation.common import validate_arch_for_backend

    with _cuda13_env((10, 0)), _fake_nccl_version(nccl_ver):
        validate_arch_for_backend("nccl_ep")


def test_nccl_floor_not_checked_on_hopper_or_for_nixl():
    from flashinfer.moe_ep.core.validation.common import validate_arch_for_backend

    # Hopper: the floor only exists for Blackwell group-create.
    with _cuda13_env((9, 0)), _fake_nccl_version((2, 27, 0)):
        validate_arch_for_backend("nccl_ep")
    # nixl_ep: does not load libnccl at all.
    with _cuda13_env((10, 0)), _fake_nccl_version((2, 27, 0)):
        validate_arch_for_backend("nixl_ep")


def test_installed_nccl_version_prefers_wheel_metadata():
    from flashinfer.moe_ep.core.validation import common

    with mock.patch("importlib.metadata.version", return_value="2.30.7"):
        assert common._installed_nccl_version() == (2, 30, 7)
