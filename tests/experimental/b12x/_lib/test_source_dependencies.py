"""Source closure tests that require neither CUDA nor the b12x runtime."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "flashinfer/experimental/b12x/_lib/source_dependencies.py"
)
spec = importlib.util.spec_from_file_location("source_dependencies", SOURCE)
dependencies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dependencies)


class SourceDependencyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.write("__init__.py", "")

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    def key(self, module="b12x.kernel"):
        resolver = dependencies.SourceDependencies(self.root)
        return resolver.fingerprint(((module, resolver.module_path(module)),))

    def test_transitive_edits_and_unrelated_files(self):
        self.write("kernel.py", "from .helper import run\n")
        self.write("helper.py", "from .leaf import VALUE\ndef run(): return VALUE\n")
        self.write("leaf.py", "VALUE = 1\n")
        before = self.key()
        self.write("README.md", "documentation")
        self.write("unrelated.py", "VALUE = 3\n")
        self.assertEqual(before, self.key())
        self.write("leaf.py", "VALUE = 2\n")
        self.assertNotEqual(before, self.key())

    def test_reexports_aliases_cycles_and_nested_imports(self):
        self.write("kernel.py", "from .helpers import run as alias\n")
        self.write("helpers/__init__.py", "from .implementation import run\n")
        self.write(
            "helpers/implementation.py",
            "def run():\n from .. import kernel\n from ..leaf import VALUE\n return VALUE\n",
        )
        self.write("leaf.py", "VALUE=1\n")
        before = self.key()
        self.write("leaf.py", "VALUE=2\n")
        self.assertNotEqual(before, self.key())

    def test_type_checking_only_import_is_excluded(self):
        self.write(
            "kernel.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n from .absent import Thing\n",
        )
        self.key()

    def test_runtime_branches_both_participate(self):
        self.write(
            "kernel.py",
            "if CHOICE:\n from .a import value\nelse:\n from .b import value\n",
        )
        self.write("a.py", "value=1\n")
        self.write("b.py", "value=2\n")
        before = self.key()
        self.write("b.py", "value=3\n")
        self.assertNotEqual(before, self.key())

    def test_missing_import_fails(self):
        self.write("kernel.py", "from .missing import value\n")
        with self.assertRaisesRegex(ValueError, "Cannot resolve"):
            self.key()

    def test_dynamic_imports_require_declared_targets(self):
        self.write("helper.py", "value=1\n")
        self.write(
            "kernel.py", "import importlib\nvalue=importlib.import_module(NAME)\n"
        )
        with self.assertRaisesRegex(ValueError, "__b12x_source_dependencies__"):
            self.key()
        self.write(
            "kernel.py",
            "import importlib\n__b12x_source_dependencies__ = ('b12x.helper',)\nvalue=importlib.import_module(NAME)\n",
        )
        before = self.key()
        self.write("helper.py", "value=2\n")
        self.assertNotEqual(before, self.key())

    def test_literal_dynamic_import(self):
        self.write(
            "kernel.py",
            "import importlib\nvalue=importlib.import_module('b12x.helper')\n",
        )
        self.write("helper.py", "value=1\n")
        before = self.key()
        self.write("helper.py", "value=2\n")
        self.assertNotEqual(before, self.key())

    def test_aliased_dynamic_import(self):
        self.write(
            "kernel.py",
            "def run():\n return load('b12x.helper')\nfrom importlib import import_module as load\n",
        )
        self.write("helper.py", "VALUE=1\n")
        before = self.key()
        self.write("helper.py", "VALUE=2\n")
        self.assertNotEqual(before, self.key())

    def test_package_computed_import_requires_declaration(self):
        self.write(
            "__init__.py", "import importlib\nvalue=importlib.import_module(NAME)\n"
        )
        self.write("kernel.py", "VALUE=1\n")
        with self.assertRaisesRegex(ValueError, "__b12x_source_dependencies__"):
            self.key()

    def test_checkout_and_alias_independence(self):
        self.write("kernel.py", "from .leaf import VALUE\n")
        self.write("leaf.py", "VALUE=1\n")
        before = self.key()
        self.assertEqual(before, self.key("flashinfer.experimental.b12x.kernel"))
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory)
            for path in self.root.rglob("*.py"):
                target = other / path.relative_to(self.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            resolver = dependencies.SourceDependencies(other)
            self.assertEqual(
                before, resolver.fingerprint((("b12x.kernel", other / "kernel.py"),))
            )

    def test_memo_does_not_reread_files(self):
        self.write("kernel.py", "VALUE=1\n")
        resolver = dependencies.SourceDependencies(self.root)
        roots = (("b12x.kernel", self.root / "kernel.py"),)
        before = resolver.fingerprint(roots)
        (self.root / "kernel.py").unlink()
        self.assertEqual(before, resolver.fingerprint(roots))

    def load(self, module, filename):
        spec = importlib.util.spec_from_file_location(module, self.root / filename)
        loaded = importlib.util.module_from_spec(spec)
        modules = patch.dict(sys.modules, {module: loaded})
        modules.start()
        self.addCleanup(modules.stop)
        spec.loader.exec_module(loaded)
        return loaded

    def test_inherited_callable_and_bound_method_sources(self):
        self.write("base.py", "class Base:\n def __call__(self): return 1\n")
        self.load("b12x.base", "base.py")
        self.write("kernel.py", "from .base import Base\nclass Kernel(Base):\n pass\n")
        target = self.load("b12x.kernel", "kernel.py").Kernel()
        for value in (target, target.__call__):
            roots = dependencies.callable_sources(value, self.root)
            self.assertEqual({name for name, _ in roots}, {"b12x.kernel", "b12x.base"})

    def test_lazy_reexport_need_not_cache_its_attribute(self):
        self.write("helpers/leaf.py", "def run(): return 1\n")
        self.load("b12x.helpers.leaf", "helpers/leaf.py")
        self.write(
            "helpers/__init__.py",
            "from importlib import import_module\ndef __getattr__(name):\n return getattr(import_module('.leaf', __name__), name)\n",
        )
        self.load("b12x.helpers", "helpers/__init__.py")
        self.write("kernel.py", "from .helpers import run as execute\n")
        self.load("b12x.kernel", "kernel.py")
        before = self.key()
        self.write("helpers/leaf.py", "def run(): return 2\n")
        self.assertNotEqual(before, self.key())

    def test_decorator_and_callable_default_sources(self):
        self.write("helper.py", "def helper(): return 1\n")
        helper = self.load("b12x.helper", "helper.py").helper
        self.write(
            "decorators.py",
            "from functools import wraps\ndef decorate(f):\n @wraps(f)\n def wrapper(*args): return f(*args)\n return wrapper\n",
        )
        decorate = self.load("b12x.decorators", "decorators.py").decorate
        self.write("kernel.py", "def kernel(callback): return callback()\n")
        kernel = self.load("b12x.kernel", "kernel.py").kernel
        kernel.__defaults__ = (helper,)
        roots = dependencies.callable_sources(decorate(kernel), self.root)
        self.assertEqual(
            {name for name, _ in roots},
            {"b12x.kernel", "b12x.helper", "b12x.decorators"},
        )

    def test_parent_initializer_dependencies(self):
        self.write("package/__init__.py", "from ..helper import VALUE\n")
        path = self.write("package/kernel.py", "VALUE=1\n")
        self.write("helper.py", "VALUE=1\n")
        roots = (("b12x.package.kernel", path),)
        before = dependencies.SourceDependencies(self.root).fingerprint(roots)
        self.write("helper.py", "VALUE=2\n")
        self.assertNotEqual(
            before, dependencies.SourceDependencies(self.root).fingerprint(roots)
        )


if __name__ == "__main__":
    unittest.main()
