"""
Copyright (c) 2024 by FlashInfer team.

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

import pathlib

# use pathlib
FLASHINFER_WORKSPACE_DIR = pathlib.Path.home() / ".flashinfer"
FLASHINFER_JIT_DIR = FLASHINFER_WORKSPACE_DIR / "cached_ops"
FLASHINFER_GEN_SRC_DIR = FLASHINFER_WORKSPACE_DIR / "generated"
_project_root = pathlib.Path(__file__).resolve().parent.parent.parent
FLASHINFER_INCLUDE_DIR = _project_root / "include"
FLASHINFER_CSRC_DIR = _project_root / "csrc"
CUTLASS_INCLUDE_DIR = _project_root / "3rdparty" / "cutlass" / "include"
