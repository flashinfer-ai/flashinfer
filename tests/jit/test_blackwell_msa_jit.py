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

from pathlib import Path

import pytest

from flashinfer.jit import blackwell_msa as loader


def _routes(target):
    return loader.ROUTES[target]


@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_every_route_names_a_program_built_for_its_target(target):
    arch = loader._TARGET_ARCH[target]
    for key, name in _routes(target).items():
        assert arch in loader.MODULES[name]["arches"], (key, name, target)
        assert loader.route_program(key, target) == name


@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_jit_spec_is_target_specific(target):
    name = next(iter(_routes(target).values()))
    spec = loader.gen_blackwell_msa_module(name, target)
    assert spec.name == f"blackwell_msa_{name}_{target}"
    assert loader.gen_blackwell_msa_module(name, target) is spec


def test_unbuilt_target_is_rejected():
    for name, record in loader.MODULES.items():
        for target, arch in loader._TARGET_ARCH.items():
            if arch not in record["arches"]:
                with pytest.raises(ValueError):
                    loader.gen_blackwell_msa_module(name, target)


def test_jit_package_reexports_resolve():
    """Every name ``flashinfer/jit/__init__.py`` imports from the loader exists."""
    import ast
    import inspect

    import flashinfer.jit

    tree = ast.parse(Path(inspect.getsourcefile(flashinfer.jit)).read_text())
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "blackwell_msa"
        for alias in node.names
    }
    assert imported, "flashinfer.jit no longer re-exports the Blackwell MSA loader"
    missing = sorted(name for name in imported if not hasattr(loader, name))
    assert not missing, missing


@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_variant_registry_lists_every_routed_program(target):
    variants = loader.BLACKWELL_MSA_VARIANTS_BY_TARGET[target]
    assert set(_routes(target).values()) <= set(variants)
    assert all(
        loader._TARGET_ARCH[target] in loader.MODULES[name]["arches"]
        for name in variants
    )
