"""Prepared BF16/FP8 paged-cache writes independent of the attention kernel.

Declare ``plan(Caps(...))``, admit it through ``PreparationSession``, then call
``run(plan, key=..., value=..., k_cache=..., v_cache=..., slot_mapping=...)``.
The caller owns the cache and graph-stable source and slot-mapping buffers.
"""

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="paged_kv_write",
    group="attention",
    api_style="planned",
    entry_points=("Caps", "Plan", "plan", "run", "is_supported", "clear_caches"),
    dtypes=("bf16", "fp8_e4m3"),
    provenance=Provenance(
        repo="https://github.com/flashinfer-ai/flashinfer",
        commit="98fb87a461183f98",
        paths=("flashinfer/experimental/b12x/attention/paged_decode/_kv_write.py",),
    ),
    test_path="tests/experimental/b12x/attention/test_paged_kv_write.py",
    notes="Prepared cache writes for vector-aligned head dimensions, including 256.",
)

if TYPE_CHECKING:
    from .api import Caps, Plan, clear_caches, is_supported, plan, run

install_lazy_api(globals(), META)
