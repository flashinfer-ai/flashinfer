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

# Former import path of ``flashinfer.experimental.dense_projection_det_gemm``: the package was renamed.  Its modules
# keep answering under this path as the renamed package's own module objects (one registry, one set of caches and
# loaded libraries), so a former import names the same objects.
import importlib
import sys

from flashinfer.experimental.dense_projection_det_gemm import *  # noqa: F401,F403

for _name in ("cake_backend", "cake_jit", "cake_build"):
    _module = importlib.import_module(f"flashinfer.experimental.dense_projection_det_gemm.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module
del _name, _module
