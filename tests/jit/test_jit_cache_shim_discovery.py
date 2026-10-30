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

import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest

from flashinfer.jit import env as jit_env


def test_shim_api_requires_version_and_provider_hook():
    """Only a module with both shim attributes is accepted."""
    assert not jit_env._is_jit_cache_shim(SimpleNamespace())
    assert not jit_env._is_jit_cache_shim(
        SimpleNamespace(__version__="0.7.2.post1+cu130")
    )
    assert not jit_env._is_jit_cache_shim(
        SimpleNamespace(get_jit_cache_providers=lambda: ())
    )
    assert jit_env._is_jit_cache_shim(
        SimpleNamespace(
            __version__="0.7.2.post1+cu130", get_jit_cache_providers=lambda: ()
        )
    )


def test_provider_only_namespace_is_ignored(monkeypatch, caplog):
    """A provider wheel leaves ``flashinfer_jit_cache/providers/`` without the shim API.

    Discovery must treat that as "no AOT providers" instead of raising AttributeError,
    which used to break ``import flashinfer`` (gh-8024).
    """
    namespace_only = types.ModuleType("flashinfer_jit_cache")
    monkeypatch.setitem(sys.modules, "flashinfer_jit_cache", namespace_only)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, package=None: object()
    )

    with caplog.at_level("WARNING", logger="flashinfer.jit"):
        assert jit_env._get_aot_providers() == ()

    assert "Ignoring flashinfer_jit_cache" in caplog.text


def test_has_flashinfer_jit_cache_matches_shim_api():
    """The public predicate agrees with the shim API check when a shim is installed."""
    if not jit_env.has_flashinfer_jit_cache():
        pytest.skip("no flashinfer-jit-cache shim installed")

    import flashinfer_jit_cache

    assert jit_env._is_jit_cache_shim(flashinfer_jit_cache)
