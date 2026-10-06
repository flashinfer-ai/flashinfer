"""Declaration-scoped host controls shared by paged planning and lowering."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import os

_NAMES = (
    "B12X_TURBO_ATTN", "B12X_PAGED_KV_TMA_PLANE_SWIZZLE", "B12X_PAGED_KV_TMA",
    "B12X_PAGED_KV_TMA_FORCE_PAGE0", "B12X_PAGED_KV_TMA_DEBUG_DUMP",
    "B12X_PAGED_KV_DEBUG_DUMP", "B12X_PAGED_EXTEND_QWEN_FP8_QK_QUARTER_REPACK",
    "B12X_PAGED_EXTEND_QWEN_FP8_PV_REPACK", "B12X_PAGED_EXTEND_COMPACT_MASK",
    "B12X_PAGED_BF16_MINIMAX_ROLE_SPECIALIZED", "B12X_PAGED_LAGUNA_DECODE_N128",
    "B12X_PAGED_LAGUNA_HEAD_PAIR_DECODE", "B12X_PAGED_LAGUNA_HEAD_PAIR_WIDE_TMA",
    "B12X_PAGED_LAGUNA_HEAD_PAIR_WIDE_HEADLOCAL", "B12X_PAGED_GQA6_COMPACT_SYNC",
    "B12X_PAGED_EXTEND_BF16_N32", "B12X_PAGED_EXTEND_BF16_N16", "B12X_PAGED_MSA",
    "B12X_PAGED_MSA_UNION_PREFILL", "B12X_PAGED_DECODE_FP8_PV_M16N16_B8",
    "B12X_PAGED_LAGUNA_HEAD_PAIR_WIDE_NOSWIZZLE",
    "B12X_PAGED_LAGUNA_DEBUG_PRINT_HEAD_PAIR_LAYOUT", "B12X_DEBUG_BF16_EXTEND_DIRECT_STORE",
    "B12X_PAGED_DECODE_GRAPH_CHUNK_PAGES", "B12X_PAGED_DECODE_GRAPH_MIN_CHUNK_PAGES",
    "B12X_DEBUG_PAGED_POLICY",
)
_CURRENT: ContextVar[Mapping[str, str | None] | None] = ContextVar("paged_controls", default=None)


def snapshot_paged_controls() -> dict[str, str | None]:
    # None and an explicitly empty environment value have distinct semantics.
    return {name: os.environ.get(name) for name in _NAMES}


def paged_control(name: str, default=None):
    controls = _CURRENT.get()
    if controls is None:
        return os.environ.get(name, default)
    value = controls.get(name)
    return default if value is None else value


@contextmanager
def paged_controls(controls: Mapping[str, str | None]):
    token = _CURRENT.set(controls)
    try:
        yield
    finally:
        _CURRENT.reset(token)
