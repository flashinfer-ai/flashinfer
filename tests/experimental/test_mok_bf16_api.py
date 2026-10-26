# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer public-API checks of :mod:`flashinfer.mok`, the MoK wrapper.

Covers what the wrapper adds over the package: the experimental marking of
its three entry points, signatures that mirror the package API (so the
documented defaults are the package defaults), lazily re-exported tables and
resolvers that are the package objects, and argument forwarding (the
package's own argument errors surface unchanged). The kernels, the adapter,
the host tables and the configuration switches are covered by the backend
suites (``test_mok_bf16.py``, ``test_mok_mxfp8.py`` and the torchrun
suites), which import the package only through
:func:`tests.test_helpers.cake_package.cake_package`. The argument-forwarding
checks call the workspace factory, which resolves the current CUDA device, so
a CUDA device is required.
"""

import inspect

import pytest

from flashinfer import mok
from flashinfer.experimental.cake_mok_bf16 import api

ENTRY_POINTS = (
    "prepare_mok_bf16",
    "create_mok_bf16_workspace",
    "quantize_mok_mxfp8_weights",
)
RE_EXPORTS = (
    "COMM_SMS",
    "IMBALANCE_COMM_SMS",
    "IMBALANCE_THRESHOLD",
    "IMBALANCE_THRESHOLDS",
    "MOK_PRECISIONS",
    "resolve_comm_sms",
    "resolve_imbalance_comm_sms",
    "resolve_imbalance_threshold",
)


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_entry_points_are_experimental_and_mirror_the_package(name):
    """Each wrapper entry point carries the experimental marking and docstring
    banner and has the package function's signature, parameter defaults
    included."""
    wrapper, implementation = getattr(mok, name), getattr(api, name)
    assert wrapper.is_experimental is True
    assert wrapper.experimental_feature == name
    assert wrapper.__doc__.startswith(".. warning::")
    assert inspect.signature(wrapper) == inspect.signature(implementation)


def test_re_exports_are_the_package_objects():
    """The communication-SM tables and resolvers are reachable from the wrapper,
    are the package objects themselves and appear in ``dir()``; unknown names
    raise ``AttributeError``."""
    for name in RE_EXPORTS:
        assert getattr(mok, name) is getattr(api, name), name
    assert set(RE_EXPORTS) <= set(dir(mok))
    assert not hasattr(mok, "resolve_nothing")  # ``__getattr__`` raises AttributeError


def test_argument_errors_of_the_package_surface_unchanged():
    """The wrapper forwards its keyword arguments to the package and raises the
    package's argument errors before any device or collective work."""
    kwargs = dict(
        group=None, device=None, num_local_tokens=512, hidden_size=256, topk=2
    )
    with pytest.raises(ValueError, match='"auto"'):
        mok.create_mok_bf16_workspace(imbalance_comm_sms="fast", **kwargs)
    with pytest.raises(ValueError, match='"auto"'):
        mok.create_mok_bf16_workspace(imbalance_threshold="fast", **kwargs)
    with pytest.raises(ValueError, match="pair"):
        mok.create_mok_bf16_workspace(imbalance_comm_sms=(24,), **kwargs)
    with pytest.raises(ValueError, match="precision must be one of"):
        mok.create_mok_bf16_workspace(precision="fp8", **kwargs)
    with pytest.raises(ValueError, match='both be "auto"'):
        mok.create_mok_bf16_workspace(
            fwd_num_comm_sms="auto", bwd_num_comm_sms=40, **kwargs
        )
    with pytest.raises(ValueError, match="Unsupported exported"):
        mok.prepare_mok_bf16(ep_size=3, local_experts=4, topk=2)
    with pytest.raises(ValueError, match="precision must be one of"):
        mok.resolve_imbalance_threshold("fp8")
