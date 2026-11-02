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

CPU-only tests for which sources the strict CuTe-DSL cache key covers
(FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY). The key must change whenever code a
kernel can run changes, and should not change when a package initializer only
gains unrelated re-exports: an overly wide key turns every new public API into
a cache invalidation for every kernel in CI's persistent compile cache.
"""

from pathlib import Path

import pytest

from flashinfer.jit import _cute_dsl_source as source_analysis


def _clear_closure_caches():
    source_analysis._import_closure.cache_clear()


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    """A tiny ``flashinfer`` tree whose initializers re-export modules."""
    pkg = tmp_path / "flashinfer"
    files = {
        # The public API: re-exports every top-level module, like the real one.
        "__init__.py": (
            '"""Public API."""\n'
            "from .api_a import api_a\n"
            "from .api_b import api_b\n"
            '__all__ = ["api_a", "api_b"]\n'
        ),
        "api_a.py": "def api_a():\n    return 1\n",
        "api_b.py": "def api_b():\n    return 2\n",
        # Mirrors flashinfer/jit/__init__.py: re-exports a submodule under its
        # own name next to unrelated module generators.
        "jit/__init__.py": (
            "from . import env as env\nfrom .generators import gen_everything\n"
        ),
        "jit/env.py": "JIT_DIR = 'cache'\n",
        "jit/generators.py": "def gen_everything():\n    return None\n",
        # A local variable that happens to share a submodule's name.
        "helpers/__init__.py": (
            "from . import heavy\n\ndef configure():\n    scale = 2\n    return scale\n"
        ),
        "helpers/scale.py": "SCALE = 1\n",
        "helpers/heavy.py": "HEAVY = 1\n",
        "kernels/__init__.py": "",
        "kernels/uses_env.py": "from ..jit import env\n",
        "kernels/uses_scale.py": "from ..helpers import scale\n",
        # Mirrors the trace templates' function-local public-API imports.
        "kernels/uses_root_api.py": (
            "def reference():\n    from flashinfer import api_a\n    return api_a()\n"
        ),
        "kernels/standalone.py": "TILE = 128\n",
    }
    for rel, text in files.items():
        _write(pkg / rel, text)
    monkeypatch.setattr(source_analysis, "_PACKAGE_DIR", pkg.resolve())
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY", "1")
    _clear_closure_caches()
    yield pkg.resolve()
    _clear_closure_caches()


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _closure_names(pkg, rel):
    closure = source_analysis._package_import_closure(pkg / rel)
    return {str(p.relative_to(pkg)) for p in closure}


def _edit_changes_key(pkg, kernel_rel, edited_rel, new_text):
    kernel = str(pkg / kernel_rel)
    _clear_closure_caches()
    before = source_analysis._hash_source_files((kernel,))
    _write(pkg / edited_rel, new_text)
    _clear_closure_caches()
    return source_analysis._hash_source_files((kernel,)) != before


# Edits that cannot change what a kernel runs must leave its key alone.


def test_self_reexport_of_submodule_does_not_follow_initializer(fake_package):
    # ``from . import env as env`` binds the very module ``from ..jit import env``
    # loads, so it cannot shadow it; following jit/__init__.py keys the kernel
    # on every generator it re-exports.
    assert "jit/generators.py" not in _closure_names(
        fake_package, "kernels/uses_env.py"
    )
    assert not _edit_changes_key(
        fake_package,
        "kernels/uses_env.py",
        "jit/generators.py",
        "def gen_everything():\n    return 1\n",
    )


def test_function_local_name_in_initializer_is_not_an_export(fake_package):
    # ``scale = 2`` inside configure() is not a package attribute.
    assert "helpers/heavy.py" not in _closure_names(
        fake_package, "kernels/uses_scale.py"
    )


def test_root_package_import_does_not_follow_whole_public_api(fake_package):
    # ``from flashinfer import api_a`` must not key the kernel on api_b.py
    # (in the real tree: decode.py, prefill.py, attention/**, ...).
    assert "api_b.py" not in _closure_names(fake_package, "kernels/uses_root_api.py")
    assert not _edit_changes_key(
        fake_package,
        "kernels/uses_root_api.py",
        "api_b.py",
        "def api_b():\n    return 3\n",
    )


def test_new_public_export_does_not_invalidate_unrelated_kernel(fake_package):
    # Every new public API edits flashinfer/__init__.py: an import line, an
    # ``__all__`` entry, perhaps the docstring. None of that is observable by a
    # kernel that does not import the new name.
    assert not _edit_changes_key(
        fake_package,
        "kernels/standalone.py",
        "__init__.py",
        '"""Public API, now with api_c."""\n'
        "from .api_a import api_a\n"
        "from .api_b import api_b\n"
        "from .api_b import api_b as api_c\n"
        '__all__ = ["api_a", "api_b"] + ["api_c"]\n',
    )


# Edits that can change what a kernel runs must change its key.


def test_initializer_code_run_at_import_is_keyed(fake_package):
    assert _edit_changes_key(
        fake_package,
        "kernels/standalone.py",
        "kernels/__init__.py",
        "TILE_OVERRIDE = 256\n",
    )
    _write(fake_package / "setup.py", "def configure():\n    return None\n")
    assert _edit_changes_key(
        fake_package,
        "kernels/standalone.py",
        "__init__.py",
        "from .api_a import api_a\nfrom .setup import configure\nconfigure()\n",
    )
    # configure() runs whenever any flashinfer module is imported.
    assert _edit_changes_key(
        fake_package,
        "kernels/standalone.py",
        "setup.py",
        "def configure():\n    return 1\n",
    )


def test_rebinding_a_used_export_is_keyed(fake_package):
    kernel = "kernels/uses_root_api.py"
    # Same source file, different object.
    assert _edit_changes_key(
        fake_package,
        kernel,
        "__init__.py",
        "from .api_b import api_b as api_a\nfrom .api_b import api_b\n",
    )
    # And the new source is followed.
    assert _edit_changes_key(
        fake_package, kernel, "api_b.py", "def api_b():\n    return 4\n"
    )


def test_package_export_shadowing_a_submodule_is_keyed(fake_package):
    _write(fake_package / "helpers/impl.py", "def scale():\n    return 2\n")
    assert _edit_changes_key(
        fake_package,
        "kernels/uses_scale.py",
        "helpers/__init__.py",
        "from .impl import scale\n",
    )
    assert _edit_changes_key(
        fake_package,
        "kernels/uses_scale.py",
        "helpers/impl.py",
        "def scale():\n    return 3\n",
    )


def test_initializer_function_body_is_followed(fake_package):
    _write(fake_package / "helpers/impl.py", "def factor():\n    return 2\n")
    _write(
        fake_package / "helpers/__init__.py",
        "from .impl import factor\n\ndef helper():\n    return factor()\n",
    )
    _write(fake_package / "kernels/uses_helper.py", "from ..helpers import helper\n")
    assert _edit_changes_key(
        fake_package,
        "kernels/uses_helper.py",
        "helpers/impl.py",
        "def factor():\n    return 3\n",
    )


@pytest.mark.parametrize(
    "kernel_source",
    [
        "from flashinfer import helpers\nX = helpers.scale.SCALE\n",
        "import flashinfer.helpers.scale as s\nX = s.SCALE\n",
        "import flashinfer.helpers.scale\nX = flashinfer.helpers.scale.SCALE\n",
        "from .. import helpers\nX = getattr(helpers, 'scale')\n",
    ],
)
def test_module_attribute_access_is_keyed(fake_package, kernel_source):
    _write(fake_package / "kernels/attr_access.py", kernel_source)
    assert _edit_changes_key(
        fake_package, "kernels/attr_access.py", "helpers/scale.py", "SCALE = 9\n"
    )


@pytest.mark.parametrize(
    "initializer",
    [
        "op = None\nfrom .impl import *\n",
        "if False:\n    op = None\n"
        "def __getattr__(name):\n"
        "    from . import impl\n"
        "    return getattr(impl, name)\n",
        "op: object\n"
        "def __getattr__(name):\n"
        "    from . import impl\n"
        "    return getattr(impl, name)\n",
    ],
    ids=[
        "star-overwrites-binding",
        "conditional-lazy-fallback",
        "annotation-lazy-fallback",
    ],
)
def test_star_import_and_lazy_getattr_are_keyed(fake_package, initializer):
    _write(fake_package / "exports/impl.py", "def op():\n    return 1\n")
    _write(fake_package / "exports/__init__.py", initializer)
    _write(fake_package / "kernels/uses_export.py", "from ..exports import op\n")
    assert _edit_changes_key(
        fake_package,
        "kernels/uses_export.py",
        "exports/impl.py",
        "def op():\n    return 2\n",
    )


@pytest.mark.parametrize(
    "signature",
    [
        "unused(arg: configure(), /)",
        "unused(arg: configure())",
        "unused(*, arg: configure())",
        "unused(*args: configure())",
        "unused(**kwargs: configure())",
        "unused() -> configure()",
    ],
    ids=[
        "positional-only",
        "positional",
        "keyword-only",
        "varargs",
        "kwargs",
        "return",
    ],
)
def test_initializer_evaluated_annotations_are_keyed(fake_package, signature):
    _write(fake_package / "setup.py", "def configure():\n    return int\n")
    _write(
        fake_package / "__init__.py",
        f"from .setup import configure\ndef {signature}:\n    pass\n",
    )
    assert _edit_changes_key(
        fake_package,
        "kernels/standalone.py",
        "setup.py",
        "def configure():\n    return str\n",
    )


def test_initializer_postponed_annotations_do_not_run_at_import(fake_package):
    _write(fake_package / "setup.py", "def configure():\n    return int\n")
    _write(
        fake_package / "__init__.py",
        "from __future__ import annotations\n"
        "from .setup import configure\n"
        "def unused(arg: configure()) -> configure():\n    pass\n",
    )
    assert not _edit_changes_key(
        fake_package,
        "kernels/standalone.py",
        "setup.py",
        "def configure():\n    return str\n",
    )


def test_conditional_bindings_key_every_branch(fake_package):
    _write(fake_package / "ops/fast.py", "def op():\n    return 1\n")
    _write(fake_package / "ops/slow.py", "def op():\n    return 1\n")
    _write(
        fake_package / "ops/__init__.py",
        "try:\n"
        "    from .fast import op\n"
        "except ImportError:\n"
        "    from .slow import op\n",
    )
    _write(fake_package / "kernels/uses_op.py", "from ..ops import op\n")
    for branch in ("ops/fast.py", "ops/slow.py"):
        assert _edit_changes_key(
            fake_package, "kernels/uses_op.py", branch, "def op():\n    return 2\n"
        )


def test_reading_all_through_star_import_keys_it(fake_package):
    _write(fake_package / "kernels/uses_all.py", "from flashinfer import *\n")
    assert _edit_changes_key(
        fake_package,
        "kernels/uses_all.py",
        "__init__.py",
        "from .api_a import api_a\nfrom .api_b import api_b\n__all__ = ['api_a']\n",
    )


# Real-tree guards. Plain-module closure sizes on main before this rework:
# cute_dsl_core 7, gdn_decode_mtp 12, nvfp4_quantize 22. Every kernel now also
# keys its package initializers and the Trace Apply code flashinfer/__init__.py
# runs at import when FLASHINFER_TRACE_APPLY is set (about 20 files).
_FLASHINFER_DIR = Path(source_analysis.__file__).resolve().parents[1]
_REAL_KERNELS = [
    "jit/cute_dsl_core.py",
    "gdn_kernels/gdn_decode_mtp.py",
    "quantization/kernels/nvfp4_quantize.py",
    "gemm/gemm_svdquant.py",
]


@pytest.fixture
def real_package(monkeypatch):
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY", "1")
    _clear_closure_caches()
    yield _FLASHINFER_DIR
    _clear_closure_caches()


@pytest.mark.parametrize(
    "kernel, limit",
    [
        ("jit/cute_dsl_core.py", 35),
        ("gdn_kernels/gdn_decode_mtp.py", 45),
        ("quantization/kernels/nvfp4_quantize.py", 50),
    ],
)
def test_real_kernel_closure_stays_near_its_imports(real_package, kernel, limit):
    closure = source_analysis._package_import_closure(real_package / kernel)
    assert len(closure) <= limit, sorted(
        str(p.relative_to(real_package)) for p in closure
    )


@pytest.mark.parametrize("kernel", _REAL_KERNELS)
def test_real_public_api_gaining_an_export_keeps_keys(real_package, kernel, tmp_path):
    files, initializers = source_analysis._key_sources((str(real_package / kernel),))
    root = str(real_package / "__init__.py")
    assert root not in files  # keyed by the bindings used, not by content
    used = initializers[root]
    edited = tmp_path / "__init__.py"
    edited.write_text(
        (real_package / "__init__.py").read_text()
        + "\nfrom .utils import get_compute_capability as _new_public_api\n"
    )
    assert source_analysis._initializer_digest(
        Path(root), used
    ) == source_analysis._initializer_digest(edited, used)


def test_cute_dsl_core_closure_excludes_unrelated_module_generators(real_package):
    closure = source_analysis._package_import_closure(
        real_package / "jit" / "cute_dsl_core.py"
    )
    unrelated = sorted(
        str(p.relative_to(real_package))
        for p in closure
        if p.is_relative_to(real_package / "jit" / "attention")
        or p.is_relative_to(real_package / "jit" / "gemm")
    )
    assert not unrelated


def test_gemm_kernel_closure_excludes_attention_stack(real_package):
    # gemm_svdquant reaches flashinfer/__init__.py through the trace
    # templates' function-local ``from flashinfer import ...``.
    closure = source_analysis._package_import_closure(
        real_package / "gemm" / "gemm_svdquant.py"
    )
    attention = sorted(
        str(p.relative_to(real_package))
        for p in closure
        if p.name in {"decode.py", "prefill.py"}
        and p.parent == real_package
        or p.is_relative_to(real_package / "attention" / "prims_ts")
    )
    assert not attention
