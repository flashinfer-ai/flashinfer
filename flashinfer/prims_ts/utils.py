# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for the FlashInfer Prims-TS integration.
"""

import importlib.util
import os
from pathlib import Path

import torch

from flashinfer.utils import get_compute_capability


def configure_cute_dsl_cache_dir() -> str:
    """Set a stable CuTe DSL file-cache directory if the user did not choose one."""
    cache_dir = os.environ.get("CUTE_DSL_CACHE_DIR")
    if cache_dir:
        return cache_dir

    workspace_base = Path(
        os.environ.get("FLASHINFER_WORKSPACE_BASE", Path.home().as_posix())
    ).expanduser()
    cache_dir = str(workspace_base / ".cache" / "flashinfer" / "cute_dsl")
    os.environ["CUTE_DSL_CACHE_DIR"] = cache_dir
    return cache_dir


def is_prims_ts_available() -> bool:
    try:
        return (
            importlib.util.find_spec("cutlass") is not None
            and importlib.util.find_spec("cutlass.cute") is not None
            and importlib.util.find_spec("cutlass.experimental.primitives") is not None
            and importlib.util.find_spec("cutlass.experimental.task_scheduling")
            is not None
        )
    except ModuleNotFoundError:
        return False


def is_prims_ts_device_supported(device: torch.device) -> bool:
    """Return whether Prims-TS kernels support the device's CUDA architecture.

    This checks device support only. Use :func:`is_prims_ts_available` to check
    dependencies; individual kernels also validate their operation configuration.
    Non-CUDA devices and environments without CUDA return ``False``.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    major, minor = get_compute_capability(device)
    return (major, minor) in ((10, 0), (10, 3))


def get_prims_ts_compile_options() -> str:
    return os.environ.get("FLASHINFER_PRIMS_TS_COMPILE_OPTIONS", "--opt-level 2")
