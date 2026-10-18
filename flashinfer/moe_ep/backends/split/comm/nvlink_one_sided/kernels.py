"""Loader of the NVLink one-sided all-to-all kernels and their workspace layout indices."""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any


@functools.cache
def get_nvlink_one_sided_module() -> Any:
    """Build (on first use) and load the JIT module backing ``NVLinkOneSidedAlltoAll``."""
    from ......jit.comm import gen_moe_ep_nvlink_one_sided_module

    return gen_moe_ep_nvlink_one_sided_module().build_and_load()


@functools.cache
def layout_constants() -> SimpleNamespace:
    """Workspace-layout metainfo indices and kernel limits exported by the module.

    Attribute names drop the ``MOE_A2A_`` prefix of the C++ constants, e.g.
    ``layout_constants().COMBINE_INPUT_OFFSET_INDEX`` or ``.MAX_RANKS``.
    """
    names, values = get_nvlink_one_sided_module().moe_a2a_get_metainfo_index_pairs()
    return SimpleNamespace(
        **{
            str(name).removeprefix("MOE_A2A_"): int(value)
            for name, value in zip(names, values, strict=True)
        }
    )
