"""Load the embedded implementation without initializing FlashInfer."""

import importlib
import importlib.abc
import importlib.metadata
import importlib.util
from pathlib import Path
import sys


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, canonical_name):
        self.canonical_name = canonical_name

    def create_module(self, spec):
        module = importlib.import_module(self.canonical_name)
        self.canonical_spec = module.__spec__
        return module

    def exec_module(self, module):
        # Import machinery sets __spec__ even when create_module reuses a module.
        module.__spec__ = self.canonical_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    prefix = "flashinfer.experimental.b12x."

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.prefix):
            return None
        canonical = "b12x." + fullname[len(self.prefix) :]
        module = importlib.import_module(canonical)
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(canonical), is_package=hasattr(module, "__path__")
        )


class _StandaloneFinder(importlib.abc.MetaPathFinder):
    """Explain imports of modules that only the standalone b12x wheel installs.

    Both wheels install b12x/__init__.py. With this package in place, modules
    such as the standalone vLLM plugins are outside the search path.
    """

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("b12x."):
            return None
        try:
            dist = importlib.metadata.distribution("b12x")
        except importlib.metadata.PackageNotFoundError:
            return None
        stem = fullname.replace(".", "/")
        files = {str(file) for file in dist.files or ()}
        if f"{stem}.py" not in files and f"{stem}/__init__.py" not in files:
            return None
        raise ModuleNotFoundError(
            f"No module named {fullname!r}: it belongs to the standalone b12x "
            f"{dist.version} distribution, but flashinfer-python provides the "
            "top-level b12x package. Uninstall b12x, then reinstall "
            "flashinfer-python: both install b12x/__init__.py, so the "
            "uninstall removes FlashInfer's copy too.",
            name=fullname,
        )


_flashinfer = importlib.util.find_spec("flashinfer")
if _flashinfer is None or _flashinfer.origin is None:
    raise ImportError("The b12x compatibility package requires flashinfer-python.")
_root = Path(_flashinfer.origin).parent / "experimental" / "b12x"
_spec = importlib.util.spec_from_file_location(
    __name__, _root / "_api.py", submodule_search_locations=[str(_root)]
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
try:
    _spec.loader.exec_module(_module)
except BaseException:
    sys.modules.pop(__name__, None)
    raise
sys.meta_path.insert(0, _AliasFinder())
# Consulted only after the regular finders fail.
sys.meta_path.append(_StandaloneFinder())
