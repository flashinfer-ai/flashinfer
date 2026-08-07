"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Tests for the experimental-feature gating primitives
# (@flashinfer_experimental_api, require_experimental). CPU-only; no GPU or
# JIT compilation required.

import warnings

import pytest

from flashinfer.api_logging import (
    _EXPERIMENTAL_ENV_VAR,
    ExperimentalWarning,
    flashinfer_experimental_api,
    is_experimental_enabled,
    require_experimental,
)


def _make_add():
    @flashinfer_experimental_api
    def sample_add(x, y):
        """Add two values."""
        return x + y

    return sample_add


def test_gate_off_raises(monkeypatch):
    monkeypatch.delenv(_EXPERIMENTAL_ENV_VAR, raising=False)
    sample_add = _make_add()
    with pytest.raises(RuntimeError, match=_EXPERIMENTAL_ENV_VAR):
        sample_add(1, 2)


def test_gate_off_for_non_one_values(monkeypatch):
    for value in ("0", "", "true", "yes"):
        monkeypatch.setenv(_EXPERIMENTAL_ENV_VAR, value)
        assert not is_experimental_enabled()


def test_gate_on_runs_and_warns_once(monkeypatch):
    monkeypatch.setenv(_EXPERIMENTAL_ENV_VAR, "1")
    sample_add = _make_add()

    with pytest.warns(ExperimentalWarning, match="sample_add"):
        assert sample_add(1, 2) == 3

    # Second call must not warn again.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        assert sample_add(3, 4) == 7


def test_mechanical_identification(monkeypatch):
    sample_add = _make_add()
    assert sample_add.is_experimental is True
    assert sample_add.experimental_feature.endswith("sample_add")
    assert "experimental" in sample_add.__doc__
    # Original docstring is preserved after the banner.
    assert "Add two values." in sample_add.__doc__


def test_decorator_with_arguments(monkeypatch):
    monkeypatch.delenv(_EXPERIMENTAL_ENV_VAR, raising=False)

    @flashinfer_experimental_api(feature="my_feature")
    def sample_mul(x, y):
        return x * y

    assert sample_mul.experimental_feature == "my_feature"
    with pytest.raises(RuntimeError, match="my_feature"):
        sample_mul(2, 3)

    monkeypatch.setenv(_EXPERIMENTAL_ENV_VAR, "1")
    with pytest.warns(ExperimentalWarning, match="my_feature"):
        assert sample_mul(2, 3) == 6


def test_require_experimental(monkeypatch):
    monkeypatch.delenv(_EXPERIMENTAL_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="some backend"):
        require_experimental("some backend")

    monkeypatch.setenv(_EXPERIMENTAL_ENV_VAR, "1")
    require_experimental("some backend")  # must not raise


def test_experimental_namespace_reexports():
    import flashinfer.experimental as fe

    assert fe.flashinfer_experimental_api is flashinfer_experimental_api
    assert fe.require_experimental is require_experimental
    assert fe.is_experimental_enabled is is_experimental_enabled
    assert fe.ExperimentalWarning is ExperimentalWarning
