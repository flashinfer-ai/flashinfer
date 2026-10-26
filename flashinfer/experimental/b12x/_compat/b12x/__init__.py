"""Load the embedded implementation without initializing FlashInfer."""

import importlib
import importlib.abc
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
