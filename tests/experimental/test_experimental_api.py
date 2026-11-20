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

# Tests for the experimental-feature primitives: @flashinfer_experimental_api,
# @experimental_backend inside @backend_requirement, and the
# FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS gate on automatic selection.
# CPU-only; no GPU or JIT compilation required.

import warnings

import pytest

import flashinfer.api_logging as api_logging
from flashinfer.api_logging import (
    _EXPERIMENTAL_AUTO_ENV_VAR,
    ExperimentalWarning,
    experimental_auto_backends_allowed,
    flashinfer_experimental_api,
    require_experimental_auto_backends,
)
from flashinfer.utils import (
    BackendSupportedError,
    backend_requirement,
    experimental_backend,
)


@pytest.fixture(autouse=True)
def _reset_gate(monkeypatch):
    # Gate off and no remembered warnings, so every test starts from the same state.
    monkeypatch.delenv(_EXPERIMENTAL_AUTO_ENV_VAR, raising=False)
    monkeypatch.setattr(api_logging, "_WARNED_EXPERIMENTAL_BACKENDS", set())


def _make_add():
    @flashinfer_experimental_api
    def sample_add(x, y):
        """Add two values."""
        return x + y

    return sample_add


# --------------------------------------------------------------------------
# @flashinfer_experimental_api: calling it is the opt-in
# --------------------------------------------------------------------------


def test_experimental_api_runs_without_env_var_and_warns_once():
    sample_add = _make_add()

    with pytest.warns(ExperimentalWarning, match="sample_add"):
        assert sample_add(1, 2) == 3

    # Second call must not warn again.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        assert sample_add(3, 4) == 7


def test_decorator_with_arguments():
    @flashinfer_experimental_api(feature="my_feature")
    def sample_mul(x, y):
        return x * y

    assert sample_mul.experimental_feature == "my_feature"
    with pytest.warns(ExperimentalWarning, match="my_feature"):
        assert sample_mul(2, 3) == 6


def test_mechanical_identification():
    sample_add = _make_add()
    assert sample_add.is_experimental is True
    assert sample_add.experimental_feature.endswith("sample_add")
    assert "experimental" in sample_add.__doc__
    # The banner no longer claims an environment variable is required.
    assert _EXPERIMENTAL_AUTO_ENV_VAR not in sample_add.__doc__
    # Original docstring is preserved after the banner.
    assert "Add two values." in sample_add.__doc__


def test_original_function_is_marked_for_registry_filtering():
    def raw(x):
        return x

    decorated = flashinfer_experimental_api(raw)
    # The trace registry stores the original, so the flag must live there too.
    assert raw.is_experimental is True
    assert decorated.is_experimental is True


# --------------------------------------------------------------------------
# The environment variable gates automatic selection only
# --------------------------------------------------------------------------


def test_gate_accepts_only_one(monkeypatch):
    assert not experimental_auto_backends_allowed()
    for value in ("0", "", "true", "yes"):
        monkeypatch.setenv(_EXPERIMENTAL_AUTO_ENV_VAR, value)
        assert not experimental_auto_backends_allowed()
    monkeypatch.setenv(_EXPERIMENTAL_AUTO_ENV_VAR, "1")
    assert experimental_auto_backends_allowed()


def test_require_experimental_auto_backends(monkeypatch):
    with pytest.raises(RuntimeError, match=_EXPERIMENTAL_AUTO_ENV_VAR):
        require_experimental_auto_backends("some_api -> some backend")

    monkeypatch.setenv(_EXPERIMENTAL_AUTO_ENV_VAR, "1")
    require_experimental_auto_backends("some_api -> some backend")  # must not raise


def test_experimental_namespace_reexports():
    import flashinfer.experimental as fe

    assert fe.flashinfer_experimental_api is flashinfer_experimental_api
    assert fe.experimental_backend is experimental_backend
    assert fe.require_experimental_auto_backends is require_experimental_auto_backends
    assert fe.experimental_auto_backends_allowed is experimental_auto_backends_allowed
    assert fe.ExperimentalWarning is ExperimentalWarning


# --------------------------------------------------------------------------
# @experimental_backend inside @backend_requirement
# --------------------------------------------------------------------------


def _any_cc(checker):
    # Test inputs carry no tensor, so backend_requirement sees capability=None;
    # accept it so the CC check is not what decides these tests.
    checker.is_compute_capability_supported = lambda cc: True
    return checker


def _make_api(*, prefer_experimental=True, stable=True):
    @_any_cc
    def _check_stable(x, backend="auto"):
        return True

    @experimental_backend
    @_any_cc
    def _check_risky(x, backend="auto"):
        return True

    def _heuristic(backends, *args, **kwargs):
        order = ["risky", "stable"] if prefer_experimental else ["stable", "risky"]
        return [b for b in order if b in backends]

    checks = {"risky": _check_risky}
    if stable:
        checks["stable"] = _check_stable

    @backend_requirement(checks, heuristic_func=_heuristic)
    def api(x, backend="auto"):
        if backend == "auto":
            backend = api.suitable_auto_backends[0]
        return backend

    return api


def test_marker_sets_flag_and_is_listed():
    api = _make_api()
    assert api.experimental_backends == frozenset({"risky"})
    assert api.is_backend_supported("risky") is True  # explicit use is allowed


def test_auto_excludes_experimental_backend_when_gate_off():
    api = _make_api()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        assert api(1) == "stable"
    assert api.suitable_auto_backends == ["stable"]
    assert api.dropped_experimental_backends == ["risky"]


def test_auto_includes_experimental_backend_when_gate_on(monkeypatch):
    monkeypatch.setenv(_EXPERIMENTAL_AUTO_ENV_VAR, "1")
    api = _make_api()
    with pytest.warns(ExperimentalWarning, match="risky.*automatically"):
        assert api(1) == "risky"
    assert api.suitable_auto_backends == ["risky", "stable"]
    # Once per (api, backend) pair.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        assert api(1) == "risky"


def test_auto_with_only_experimental_candidates_hints_at_env_var():
    api = _make_api(stable=False)
    with pytest.raises(BackendSupportedError, match=_EXPERIMENTAL_AUTO_ENV_VAR):
        api(1)


def test_common_check_failure_does_not_reuse_a_stale_experimental_hint():
    """A common_check failure must not inherit the previous call's dropped list.

    suitable_auto_backends returns early when the common check fails, so the
    attribute the hint reads has to be cleared before that return -- otherwise a
    problem-size failure is reported as an experimental-backend exclusion and
    points the user at an environment variable that will not help.
    """
    allowed = {"value": True}

    def _common(x, backend="auto"):
        return allowed["value"]

    @_any_cc
    def _check_stable(x, backend="auto"):
        return True

    @experimental_backend
    @_any_cc
    def _check_risky(x, backend="auto"):
        return True

    @backend_requirement(
        {"stable": _check_stable, "risky": _check_risky},
        common_check=_common,
        heuristic_func=lambda backends, *args, **kwargs: list(backends),
    )
    def api(x, backend="auto"):
        return api.suitable_auto_backends[0] if backend == "auto" else backend

    # First call succeeds and records "risky" as dropped from automatic selection.
    assert api(1) == "stable"
    assert api.dropped_experimental_backends == ["risky"]

    # Now the common check is what fails. The experimental gate is irrelevant to
    # that, so the hint must not appear.
    allowed["value"] = False
    with pytest.raises(BackendSupportedError) as excinfo:
        api(1)
    assert _EXPERIMENTAL_AUTO_ENV_VAR not in str(excinfo.value)
    assert api.dropped_experimental_backends == []


def test_explicit_experimental_backend_needs_no_env_var_and_warns_once():
    api = _make_api()
    with pytest.warns(ExperimentalWarning, match="risky.*selected for 'api'"):
        assert api(1, backend="risky") == "risky"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        assert api(1, backend="risky") == "risky"
        assert api(1, backend="risky", skip_check=True) == "risky"
        assert api(1, backend="stable") == "stable"  # stable never warns


def test_backstop_rejects_unmarked_checker_under_experimental_package():
    @_any_cc
    def _check_unmarked(x, backend="auto"):
        return True

    _check_unmarked.__module__ = "flashinfer.experimental.fake.support"

    with pytest.raises(ValueError, match="not marked @experimental_backend"):

        @backend_requirement({"fake": _check_unmarked})
        def api(x, backend="fake"):
            return backend

    # Marked, the same checker is accepted.
    experimental_backend(_check_unmarked)

    @backend_requirement({"fake": _check_unmarked})
    def api_ok(x, backend="fake"):
        return backend

    assert api_ok.experimental_backends == frozenset({"fake"})
