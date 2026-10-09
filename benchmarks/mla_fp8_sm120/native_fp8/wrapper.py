"""Load the checkout backend without replacing the installed FlashInfer baseline."""

import importlib.util
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[3] / "flashinfer/experimental/mla_fp8_sm120"
NAME = "_flashinfer_mla_fp8_sm120_study"
if NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(NAME, BACKEND / "__init__.py")
    package = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = package
    spec.loader.exec_module(package)
spec = importlib.util.spec_from_file_location(NAME + ".wrapper", BACKEND / "wrapper.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
NativeMLA = module.NativeMLA
build = module.build

if __name__ == "__main__":
    print(build())
