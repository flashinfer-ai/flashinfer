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

from ._core import *  # noqa: F401,F403


_PRIMS_TS_LAZY_EXPORTS = frozenset(
    {
        "get_prims_ts_batch_decode_mla_workspace_size",
        "prims_ts_batch_decode_with_kv_cache_mla",
    }
)

_SPARSE_MLA_SM120_LAZY_EXPORTS = frozenset(
    {
        "SparseMLASm120DecodeConfig",
        "supported_sparse_mla_sm120_configs",
    }
)


def __getattr__(name: str):
    """Resolve lazily-exported MLA APIs without loading their runtime at import."""

    if name in _PRIMS_TS_LAZY_EXPORTS:
        from ..attention.prims_ts import mla_decode as prims_ts_mla_decode

        value = getattr(prims_ts_mla_decode, name)
        globals()[name] = value
        return value
    if name in _SPARSE_MLA_SM120_LAZY_EXPORTS:
        from . import _sparse_mla_sm120

        value = getattr(_sparse_mla_sm120, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include lazily-exported names alongside the module globals."""
    return sorted(
        set(globals()) | _PRIMS_TS_LAZY_EXPORTS | _SPARSE_MLA_SM120_LAZY_EXPORTS
    )
