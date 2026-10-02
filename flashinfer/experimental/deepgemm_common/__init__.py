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

# Shared infrastructure of the generated DeepGEMM-family backends
# (flashinfer/experimental/deepgemm_*, mega_moe_v3, source_mega_moe): one
# catalog reader, one architecture / SM-count route lookup and one JIT loader.
# Family modules keep their own route keys, plans and launch logic.

from .cake_catalog import (
    ARCHES,
    Catalog,
    UnsupportedDevice,
    jit_spec_name,
    load_catalog,
    nvcc_flags,
)

__all__ = [
    "ARCHES",
    "Catalog",
    "UnsupportedDevice",
    "jit_spec_name",
    "load_catalog",
    "nvcc_flags",
]
