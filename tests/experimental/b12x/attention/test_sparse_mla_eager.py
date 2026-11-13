"""attention.sparse_mla_eager: public API surface and support gate.

The eager planned lifecycle (plan -> bind -> run_decode/run_extend) shares the
master sparse-MLA kernels; this module pins the entry points and the SM120
support gate without launching kernels.
"""

from __future__ import annotations

import pytest

from b12x.attention import sparse_mla_eager

from ..conftest import require_b12x


def test_sparse_mla_eager_entry_points():
    require_b12x()
    for name in (
        "Caps",
        "Plan",
        "Binding",
        "Scratch",
        "DecodeMetadata",
        "ExtendMetadata",
        "plan",
        "bind",
        "run_decode",
        "run_extend",
        "is_supported",
        "clear_caches",
    ):
        assert hasattr(sparse_mla_eager, name), f"missing entry point {name}"
    assert callable(sparse_mla_eager.plan)
    assert callable(sparse_mla_eager.bind)
    assert callable(sparse_mla_eager.run_decode)
    assert callable(sparse_mla_eager.run_extend)


def test_sparse_mla_eager_is_supported_gate():
    require_b12x()
    supported = sparse_mla_eager.is_supported()
    assert isinstance(supported, bool)
