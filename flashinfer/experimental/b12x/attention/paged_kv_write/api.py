"""Standalone preparation and execution of paged K/V cache writes."""

from dataclasses import dataclass

import torch

from b12x._lib.gating import default_is_supported
from b12x.preparation import Plan
from b12x.preparation.types import plan_from_handle, require_prepared

from . import META


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Static cache geometry; token capacity comes from the bound slot map."""

    device: torch.device | str
    num_kv_heads: int
    head_dim_k: int
    head_dim_v: int
    page_size: int
    kv_dtype: torch.dtype = torch.bfloat16


def plan(caps: Caps, *, invocation=None, override=None) -> Plan:
    """Declare a writer; a preparation session compiles and primes its launch."""
    from ._preparation import make_plan

    return make_plan(caps, invocation=invocation, override=override)


def run(
    plan: Plan,
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Write BF16 rows to caller-owned paged caches without allocating.

    Sources are ``[T, Hkv, Dk/Dv]`` with unit inner strides; caches are
    ``[pages, page_size, Hkv, Dk/Dv]``. Addresses and outer byte strides must
    be 16-byte aligned. Nonoverlapping strided views are supported.
    ``slot_mapping`` is contiguous int64 ``[T]`` containing
    ``page * page_size + offset``. Negative and out-of-capacity destinations
    are skipped; live destinations must be distinct. Padded source rows must
    still have valid storage. Source and destination storage must not overlap,
    and K/V destination elements must be disjoint.

    FP8 caches store saturating E4M3 ``x / scale`` with one device float32
    scale per K/V cache. Scales must be finite and positive. Their values and
    slot mappings may change between replays. BF16 copies preserve every bit.
    Device metadata values are caller invariants and are not read on the host.
    A registered mutating operator retains cache writes under full-graph compilation.
    """
    _write_op(plan.handle, key, value, k_cache, v_cache, slot_mapping, k_scale, v_scale)


@torch.library.custom_op("b12x::paged_kv_write", mutates_args=("k_cache", "v_cache"))
def _write_op(
    plan_handle: int,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    plan = plan_from_handle(plan_handle)
    require_prepared(plan, "attention.paged_kv_write", key.device).run(
        key=key,
        value=value,
        k_cache=k_cache,
        v_cache=v_cache,
        slot_mapping=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
    )


@_write_op.register_fake
def _write_fake(
    plan_handle, key, value, k_cache, v_cache, slot_mapping, k_scale, v_scale
):
    pass


def is_supported(device=None):
    return default_is_supported(device, requires=META.requires)


def clear_caches():
    """Compiled programs are owned by preparation sessions."""


__all__ = ["Caps", "Plan", "plan", "run", "is_supported", "clear_caches"]
