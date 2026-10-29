"""Host-side gate: CUTLASS MXFP8×MXFP4 must refuse SM12x (#7179)."""

import pytest

from flashinfer.fused_moe.api import (
    CutlassMxfp8Mxfp4Config,
    SM12xMxfp8Mxfp4Config,
    _CUTLASS_MXFP8_MXFP4_ARCHS,
)
from flashinfer.fused_moe.runners import CutlassMxfp8Mxfp4Runner


def test_cutlass_mxfp8_mxfp4_arch_tuple_excludes_sm12x():
    assert _CUTLASS_MXFP8_MXFP4_ARCHS == (100, 103, 107, 110)
    assert CutlassMxfp8Mxfp4Runner._supported_archs == _CUTLASS_MXFP8_MXFP4_ARCHS
    assert 120 not in _CUTLASS_MXFP8_MXFP4_ARCHS
    assert 121 not in _CUTLASS_MXFP8_MXFP4_ARCHS


@pytest.mark.parametrize("arch", [120, 121, 89, 90, 130])
def test_cutlass_mxfp8_mxfp4_config_rejects_unsupported(arch):
    assert not CutlassMxfp8Mxfp4Config.supported(arch)


@pytest.mark.parametrize("arch", [100, 103, 107, 110])
def test_cutlass_mxfp8_mxfp4_config_keeps_supported(arch):
    assert CutlassMxfp8Mxfp4Config.supported(arch)


@pytest.mark.parametrize("arch", [120, 121])
def test_sm12x_cutedsl_remains_the_mxfp8_mxfp4_path(arch):
    assert SM12xMxfp8Mxfp4Config.supported(arch)
