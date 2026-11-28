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

# MOSCA-optimized kernels (experimental). See README.md in this directory.
#
# Layout (README.md, "Package layout"):
#   catalog.py           every MOSCA kernel in FlashInfer: API, config, base
#                        kernel, MOSCA changes, validated coverage (data only)
#   _vendor.py           vendored upstream trees: manifests and checks
#   vendor/<tree>.json   one manifest per vendored tree
#   csrc/vendor/<tree>/  vendored upstream sources (verbatim, no MOSCA changes)
#   csrc/common/         shims shared by kernels built on a vendored tree
#   csrc/<family>/       MOSCA kernels and FlashInfer-side host code of one family
#   <family>/            Python side of one kernel family
#                        (megamoe/: the moe_ep mega path)
#
# Importing this package is lightweight: no JIT, tvm-ffi or kernel module is
# loaded. Core never imports it (flashinfer/experimental/README.md, containment
# rules); importing a family package (e.g. ``megamoe``) registers its kernels.

from __future__ import annotations

from .catalog import KERNELS, Feature, MoscaKernel, ValidatedCell, get_kernel

__all__ = [
    "Feature",
    "KERNELS",
    "MoscaKernel",
    "ValidatedCell",
    "get_kernel",
]
