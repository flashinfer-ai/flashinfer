"""Public surface for caller-owned prepared RoCE runtimes."""
from __future__ import annotations
from ._preparation import plan, query_from_runtime
from ._tuning import RoceQuery
from .roce_oneshot import API_VERSION, DEFAULT_MAX_GATHER_BYTES, DEFAULT_MAX_SIZE, SUPPORTED_DTYPES, SUPPORTED_WORLD_SIZES, RoceOneshotAllReduce as AllReduce, default_gid_index, discover_hcas, is_supported
__all__ = ["API_VERSION", "AllReduce", "DEFAULT_MAX_GATHER_BYTES", "DEFAULT_MAX_SIZE", "SUPPORTED_DTYPES", "SUPPORTED_WORLD_SIZES", "RoceQuery", "plan", "query_from_runtime", "default_gid_index", "discover_hcas", "is_supported"]
