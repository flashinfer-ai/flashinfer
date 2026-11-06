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


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_every_route_names_a_program_built_for_its_target(target):
    arch = loader._TARGET_ARCH[target]
    for key, name in _routes(target).items():
        assert arch in loader.MODULES[name]["arches"], (key, name, target)
        assert loader.route_program(key, target) == name


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_jit_spec_is_target_specific(target):
    name = next(iter(_routes(target).values()))
    spec = loader.gen_blackwell_msa_module(name, target)
    assert spec.name == f"blackwell_msa_{name}_{target}"
    assert loader.gen_blackwell_msa_module(name, target) is spec


@pytest.mark.cpu_only
def test_every_record_names_a_registered_arg_plan():
    """Every program names one ``ARG_PLANS`` entry and every plan is named."""
    named = set()
    for name, record in loader.MODULES.items():
        plan = loader.ARG_PLANS[record["arg_plan"]]
        grid = [argument for kind, argument in plan if kind == "grid"]
        assert grid == ["grid_x", "grid_y", "grid_z"], name
        named.add(record["arg_plan"])
    assert named == set(loader.ARG_PLANS)


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_variant_registry_lists_every_routed_program(target):
    variants = loader.BLACKWELL_MSA_VARIANTS_BY_TARGET[target]
    assert set(_routes(target).values()) <= set(variants)
    assert all(
        loader._TARGET_ARCH[target] in loader.MODULES[name]["arches"]
        for name in variants
    )


def _shared_binding_records():
    """``MODULES`` records whose host binding is one shared per-argument-plan source."""
    shared = {}
    suffix = "_binding.cu"
    for name, record in loader.MODULES.items():
        stem = Path(record["sources"][1]).name[: -len(suffix)]
        if stem != name:
            shared[name] = record
    return shared


def _blackwell_msa_csrc():
    return Path(loader.__file__).resolve().parents[2] / "csrc"


@pytest.mark.cpu_only
def test_shared_bindings_take_each_record_from_its_compile_flags():
    """Every ``#ifndef CAKE_* / #error`` selector of a shared binding source is set
    exactly once, per target, by the record's compile flags; the kernel symbol and the
    plan match the record."""
    import re

    csrc = _blackwell_msa_csrc()
    shared = _shared_binding_records()
    assert shared, "no shared bindings registered"
    for name, record in shared.items():
        stem = Path(record["sources"][1]).name[: -len("_binding.cu")]
        assert stem == f"cake_blackwell_msa_{record['arg_plan']}", (name, stem)
        text = (csrc / record["sources"][1]).read_text()
        guards = re.findall(r"#ifndef (CAKE_\w+)\n#error", text)
        assert "CAKE_PROGRAM_KERNEL" in guards, (name, guards)
        assert "CAKE_BIND_SMEM" in guards, (name, guards)
        assert f"-DCAKE_PROGRAM_KERNEL=kernel_{name}" in record["compile_flags"], name
        per_target = record.get("target_compile_flags", {})
        for target, arch in loader._TARGET_ARCH.items():
            if arch not in record["arches"]:
                continue
            flags = list(record["compile_flags"]) + list(per_target.get(target, []))
            for macro in guards:
                hits = [f for f in flags if f.startswith(f"-D{macro}=")]
                assert len(hits) == 1, (name, target, macro, hits)


@pytest.mark.cpu_only
def test_shared_bindings_have_the_generated_form():
    """Each shared binding source carries the generated -D-selected form and no digest."""
    import re

    csrc = _blackwell_msa_csrc()
    bindings = {r["sources"][1] for r in _shared_binding_records().values()}
    assert bindings, "no shared bindings registered"
    for binding in sorted(bindings):
        text = (csrc / binding).read_text()
        assert 'extern "C" __global__ void CAKE_PROGRAM_KERNEL(' in text, binding
        assert "namespace CAKE_BINDING_NS {" in text, binding
        assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, CAKE_BINDING_NS::Run);" in text, (
            binding
        )
        assert "config.dynamicSmemBytes = CAKE_BIND_SMEM;" in text, binding
        assert not re.search(r"[0-9a-f]{20}", text), binding


@pytest.mark.cpu_only
def test_shared_bindings_serve_one_arg_plan_each():
    """Records sharing a binding source share one argument plan and no two carry the
    same full compile-flag set."""
    by_stem = {}
    for record in _shared_binding_records().values():
        stem = Path(record["sources"][1]).name[: -len("_binding.cu")]
        by_stem.setdefault(stem, []).append(record)
    assert by_stem, "no shared bindings registered"
    for stem, records in by_stem.items():
        assert len({r["arg_plan"] for r in records}) == 1, stem
        tuples = {
            (
                tuple(r["compile_flags"]),
                tuple(
                    (t, tuple(f))
                    for t, f in sorted(r.get("target_compile_flags", {}).items())
                ),
            )
            for r in records
        }
        assert len(tuples) == len(records), (stem, len(tuples), len(records))
