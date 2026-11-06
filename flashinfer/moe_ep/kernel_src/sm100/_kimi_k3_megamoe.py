"""Load the isolated Kimi source tree as a private Python package."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_root = Path(__file__).with_name("kimi-k3-megamoe")
_spec = spec_from_file_location(
    __name__, _root / "__init__.py", submodule_search_locations=[str(_root)]
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load Kimi MegaMoE from {_root}")
_module = module_from_spec(_spec)
_previous = sys.modules[__name__]
sys.modules[__name__] = _module
try:
    _spec.loader.exec_module(_module)
except BaseException:
    sys.modules[__name__] = _previous
    raise
