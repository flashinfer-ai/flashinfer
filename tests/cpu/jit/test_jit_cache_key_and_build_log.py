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

CPU-only tests for the strict CuTe-DSL cache key
(FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY) and the JIT build timing log
(FLASHINFER_JIT_BUILD_LOG) that CI's persistent compile cache relies on.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from flashinfer.jit import core as jit_core
from flashinfer.jit import cute_dsl_core
from flashinfer.jit import _cute_dsl_source as source_analysis

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    """A tiny ``flashinfer`` package tree the import walker is pointed at."""
    pkg = tmp_path / "flashinfer"
    files = {
        "__init__.py": "from . import everything\n",
        "everything.py": "X = 1\n",
        "helpers/__init__.py": "from .reexported import THING\n",
        "helpers/reexported.py": "THING = 1\n",
        "helpers/math.py": "SCALE = 2\n",
        "helpers/deep.py": "from flashinfer.helpers import math\n",
        "kernels/__init__.py": "",
        "kernels/kernel.py": (
            "import os\n"
            "from .. import version\n"
            "from ..helpers import deep\n"
            "from ..helpers import THING\n"
            "def compile():\n"
            "    from flashinfer.kernels import local_only\n"
        ),
        "kernels/local_only.py": "Y = 3\n",
        "version.py": "from . import _build_meta\n",
        "unrelated.py": "Z = 4\n",
    }
    for rel, text in files.items():
        path = pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(source_analysis, "_PACKAGE_DIR", pkg.resolve())
    source_analysis._import_closure.cache_clear()
    yield pkg.resolve()
    source_analysis._import_closure.cache_clear()


@pytest.mark.cpu_only
def test_import_closure_follows_helpers_but_not_unkeyed_files(fake_package):
    closure = source_analysis._package_import_closure(
        fake_package / "kernels" / "kernel.py"
    )
    names = {str(p.relative_to(fake_package)) for p in closure}
    assert names == {
        "__init__.py",  # hashed without following all public reexports
        "kernels/__init__.py",
        "kernels/kernel.py",
        "helpers/deep.py",
        "helpers/math.py",  # transitively, via an absolute import
        "helpers/__init__.py",  # THING is defined by the package itself
        "helpers/reexported.py",
        "kernels/local_only.py",  # function-local import
    }


@pytest.mark.cpu_only
def test_strict_key_tracks_implicit_parent_initializers(fake_package, monkeypatch):
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY", "1")
    kernel = str(fake_package / "kernels/kernel.py")
    before = cute_dsl_core._hash_source_files((kernel,))
    (fake_package / "kernels/__init__.py").write_text("TILE = 256\n")
    assert cute_dsl_core._hash_source_files((kernel,)) != before
    before = cute_dsl_core._hash_source_files((kernel,))
    (fake_package / "__init__.py").write_text("ROOT_SETTING = 1\n")
    assert cute_dsl_core._hash_source_files((kernel,)) != before


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "statement",
    [
        'import importlib; helper = importlib.import_module("flashinfer.helpers.math")',
        'from importlib import import_module as load; helper = load(".helpers.math", __package__)',
        'import importlib as il; helper = il.import_module(f"{__package__}.helpers.math")',
        'helper = __import__("flashinfer.helpers.math")',
        'helper = __import__("flashinfer.helpers", fromlist=["math"]).math',
        'helper = __import__("flashinfer.helpers", globals(), locals(), ["math"]).math',
        'helper = __import__("helpers.math", globals(), locals(), [], 1)',
        'import importlib; helper = importlib.import_module(name="flashinfer.helpers.math")',
    ],
)
def test_strict_key_tracks_resolvable_dynamic_imports(
    fake_package, monkeypatch, statement
):
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY", "1")
    kernel = fake_package / "dynamic.py"
    kernel.write_text(statement + "\n")
    before = cute_dsl_core._hash_source_files((str(kernel),))
    (fake_package / "helpers/math.py").write_text("SCALE = 64\n")
    assert cute_dsl_core._hash_source_files((str(kernel),)) != before


@pytest.mark.cpu_only
def test_strict_key_tracks_package_export_shadowing_a_submodule(
    fake_package, monkeypatch
):
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY", "1")
    (fake_package / "helpers/child.py").write_text("UNUSED = 1\n")
    (fake_package / "helpers/__init__.py").write_text("from .reexported import child\n")
    helper = fake_package / "helpers/reexported.py"
    helper.write_text("def child(): return 1\n")
    kernel = fake_package / "shadowed.py"
    kernel.write_text("from .helpers import child\n")
    before = cute_dsl_core._hash_source_files((str(kernel),))
    helper.write_text("def child(): return 2\n")
    assert cute_dsl_core._hash_source_files((str(kernel),)) != before


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "dependency", ["apache-tvm-ffi", "torch", "nvidia-cutlass-dsl-libs-cu13"]
)
def test_compiler_fingerprint_tracks_resolved_dependencies(monkeypatch, dependency):
    import importlib.metadata

    class Distribution:
        metadata = {"Name": dependency}
        version = "1.0"
        record = "build-1"
        direct_url = None

        def read_text(self, name):
            return self.record if name == "RECORD" else self.direct_url

    distribution = Distribution()
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [distribution])

    def fingerprint():
        cute_dsl_core._get_cute_dsl_version.cache_clear()
        return cute_dsl_core._get_cute_dsl_version()

    try:
        first = fingerprint()
        distribution.version = "1.1"
        second = fingerprint()
        distribution.record = "rebuilt-wheel"
        third = fingerprint()
        distribution.direct_url = '{"vcs_info":{"commit_id":"new-commit"}}'
        assert len({first, second, third, fingerprint()}) == 4
    finally:
        cute_dsl_core._get_cute_dsl_version.cache_clear()


@pytest.mark.cpu_only
def test_compiler_fingerprint_tracks_extension_abi(monkeypatch):
    import sysconfig

    def fingerprint(abi):
        monkeypatch.setattr(sysconfig, "get_config_var", lambda name: abi)
        cute_dsl_core._get_cute_dsl_version.cache_clear()
        return cute_dsl_core._get_cute_dsl_version()

    try:
        assert fingerprint("cpython-310-x86_64") != fingerprint("cpython-310d-x86_64")
    finally:
        cute_dsl_core._get_cute_dsl_version.cache_clear()


@pytest.mark.cpu_only
def test_strict_key_tracks_imported_helpers(fake_package, monkeypatch):
    kernel = str(fake_package / "kernels" / "kernel.py")
    helper = fake_package / "helpers" / "math.py"

    monkeypatch.delenv("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY", raising=False)
    loose_before = cute_dsl_core._hash_source_files((kernel,))
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY", "1")
    strict_before = cute_dsl_core._hash_source_files((kernel,))

    helper.write_text("SCALE = 3\n")
    source_analysis._import_closure.cache_clear()

    assert cute_dsl_core._hash_source_files((kernel,)) != strict_before
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY")
    assert cute_dsl_core._hash_source_files((kernel,)) == loose_before

    # Files outside the closure never affect the key.
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY", "1")
    strict_after = cute_dsl_core._hash_source_files((kernel,))
    (fake_package / "unrelated.py").write_text("Z = 5\n")
    assert cute_dsl_core._hash_source_files((kernel,)) == strict_after


@pytest.mark.cpu_only
def test_strict_key_leaves_files_outside_the_package_alone(
    fake_package, tmp_path, monkeypatch
):
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY", "1")
    outside = tmp_path / "standalone.py"
    outside.write_text("from flashinfer.helpers import math\n")
    assert source_analysis._expand_key_files((str(outside),)) == (str(outside),)


@pytest.mark.cpu_only
def test_strict_key_parse_failure_bypasses_disk_cache(fake_package, monkeypatch):
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY", "1")
    (fake_package / "helpers" / "math.py").write_text("def broken(:\n")
    result = cute_dsl_core.build_and_load_cute_dsl_kernel(
        "fake_module",
        "fake_kernel",
        lambda: "compiled in process",
        extra_key_files=(str(fake_package / "kernels" / "kernel.py"),),
    )
    assert result == "compiled in process"


@pytest.mark.cpu_only
def test_strict_closure_of_real_kernel_stays_small(monkeypatch):
    """Guards against the closure ballooning to the whole package again."""
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY", "1")
    kernel = Path(cute_dsl_core.__file__).resolve().parents[1] / "gdn_kernels"
    closure = source_analysis._package_import_closure(kernel / "gdn_decode_mtp.py")
    assert any(
        p.parent.name == "gdn_kernels" and p.name != "gdn_decode_mtp.py"
        for p in closure
    )  # picks up at least one sibling helper
    assert len(closure) < 100
    # jit/env.py imports version.py, whose git commit changes on every checkout.
    assert not {"version.py", "_build_meta.py"} & {p.name for p in closure}


class _FakeSpec(jit_core.JitSpec):
    toolchain = "fake"

    def __init__(self, tmp_path):
        self.name = "fake_module"
        self._tmp_path = tmp_path
        self.built = False

    @property
    def lock_path(self):
        return self._tmp_path / "fake.lock"

    @property
    def is_compiled(self):
        return self.built

    def get_library_path(self):
        return self._tmp_path / "fake.so"

    def try_load(self):
        return "cached" if self.built else None

    def build(self):
        self.built = True

    def load(self):
        return "built"


@pytest.mark.cpu_only
def test_build_and_load_records_build_time(tmp_path, monkeypatch):
    log = tmp_path / "builds.jsonl"
    monkeypatch.setenv("FLASHINFER_JIT_BUILD_LOG", str(log))
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    spec = _FakeSpec(tmp_path)
    assert spec.build_and_load() == "built"
    assert spec.build_and_load() == "cached"  # a cache hit is not a build

    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["toolchain"] == "fake"
    assert records[0]["name"] == "fake_module"
    assert records[0]["seconds"] >= 0


@pytest.mark.cpu_only
def test_build_log_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASHINFER_JIT_BUILD_LOG", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.chdir(tmp_path)
    _FakeSpec(tmp_path).build_and_load()
    assert not list(tmp_path.glob("*.jsonl"))


@pytest.mark.cpu_only
def test_toolchain_labels():
    assert jit_core.JitSpecNvcc.toolchain == "nvcc"
    assert cute_dsl_core.JitSpecCuteDsl.toolchain == "cute_dsl"


def _load_summarizer():
    spec = importlib.util.spec_from_file_location(
        "summarize_jit_builds", REPO_ROOT / "scripts" / "summarize_jit_builds.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu_only
def test_summary_totals_by_toolchain(tmp_path):
    log = tmp_path / "builds.jsonl"
    rows = [
        {"toolchain": "nvcc", "name": "a", "seconds": 0.05},
        {"toolchain": "nvcc", "name": "b", "seconds": 120.0},
        {"toolchain": "cute_dsl", "name": "c", "seconds": 30.0},
        {"toolchain": "cute_dsl", "name": "d", "seconds": 30.0},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n{truncated")
    summarizer = _load_summarizer()
    records = summarizer.load_records(log)
    assert len(records) == 4  # the partial trailing line is skipped

    text = summarizer.summarize(records, 1.0, 2, markdown=False)
    assert "toolchain=nvcc builds=1 total_seconds=120.0 up_to_date=1" in text
    assert "toolchain=cute_dsl builds=2 total_seconds=60.0 up_to_date=0" in text
    assert text.index("toolchain=nvcc") < text.index("toolchain=cute_dsl")

    table = summarizer.summarize(records, 1.0, 2, markdown=True)
    assert "| cute_dsl | 2 | 1.0 min | 0 |" in table
