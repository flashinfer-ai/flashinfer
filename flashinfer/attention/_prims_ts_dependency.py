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

"""Feature-local CUTLASS DSL dependency check for PrimTS attention."""

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

from packaging.version import InvalidVersion, Version

_CUTLASS_DSL_DISTRIBUTION = "nvidia-cutlass-dsl"
_MINIMUM_CUTLASS_DSL_VERSION = Version("4.7.0")


def require_prims_ts_cutlass_dsl(
    get_version: Callable[[str], str] = distribution_version,
) -> str:
    """Return the installed DSL version or reject an unsupported PrimTS import."""

    install_hint = (
        "Install nvidia-cutlass-dsl==4.7.0, or "
        "nvidia-cutlass-dsl[cu13]==4.7.0 for CUDA 13."
    )
    try:
        installed_text = get_version(_CUTLASS_DSL_DISTRIBUTION)
    except PackageNotFoundError as error:
        raise ImportError(
            "flashinfer.attention.prims_ts requires nvidia-cutlass-dsl>=4.7.0, "
            f"but it is not installed. {install_hint}"
        ) from error

    try:
        installed = Version(installed_text)
    except InvalidVersion as error:
        raise ImportError(
            "flashinfer.attention.prims_ts requires nvidia-cutlass-dsl>=4.7.0, "
            f"but found an invalid version {installed_text!r}. {install_hint}"
        ) from error

    if installed < _MINIMUM_CUTLASS_DSL_VERSION:
        raise ImportError(
            "flashinfer.attention.prims_ts requires nvidia-cutlass-dsl>=4.7.0, "
            f"but found {installed_text}. FlashInfer's default dependency remains 4.6.2. "
            f"{install_hint}"
        )
    return installed_text
