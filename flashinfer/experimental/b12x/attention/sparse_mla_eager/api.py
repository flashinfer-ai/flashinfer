"""Public surface for attention.sparse_mla (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from .._shared.mla.api import (
    MLASparseDecodeMetadata as DecodeMetadata,
)
from .._shared.mla.api import (
    MLASparseExtendMetadata as ExtendMetadata,
)
from .._shared.mla.api import (
    clear_mla_caches as clear_caches,
)
from .._shared.mla.api import (
    sparse_mla_decode_forward as _master_decode_forward,
)
from .._shared.mla.api import (
    sparse_mla_extend_forward as _master_extend_forward,
)
from b12x.attention.sparse_mla._scratch import (
    B12XSparseMLABinding as _MasterBinding,
)


def _to_master_binding(binding, kv_cache):
    """Translate a fork-era Binding into master's Binding shape.

    Master's forwards read ``kv_cache`` (and the format knobs) off the
    Binding; the fork-era Binding carries the per-call tensors only and
    the integration passes ``kv_cache`` per call. Keep that contract.
    """
    return _MasterBinding(
        scratch=binding.scratch,
        q=binding.q,
        selected_indices=binding.selected_indices,
        cache_seqlens_int32=binding.cache_seqlens_int32,
        nsa_cache_seqlens_int32=binding.nsa_cache_seqlens_int32,
        kv_cache=kv_cache,
    )


def run_decode(*, binding, kv_cache=None, **kwargs):
    return _master_decode_forward(
        binding=_to_master_binding(binding, kv_cache), **kwargs
    )


def run_extend(*, binding, kv_cache=None, **kwargs):
    return _master_extend_forward(
        binding=_to_master_binding(binding, kv_cache), **kwargs
    )


from ._scratch import (
    B12XSparseMLABinding as Binding,
)
from ._scratch import (
    B12XSparseMLAScratch as Scratch,
)
from ._scratch import (
    B12XSparseMLAScratchCaps as Caps,
)
from ._scratch import (
    B12XSparseMLAScratchPlan as Plan,
)
from ._scratch import (
    plan_sparse_mla_scratch as plan,
)
from . import META


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind runtime tensors and caller-owned scratch to a plan.

    Views only — never allocates — so it is CUDA-graph-capture safe.
    Delegates to ``plan.bind(**kwargs)``.
    """
    return plan.bind(**kwargs)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
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
]
