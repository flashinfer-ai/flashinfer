"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Hopper (SM90) variable block-sparse attention on the generated Cake kernel.

``backend="cake"`` of :class:`flashinfer.sparse.VariableBlockSparseAttentionWrapper`
routes here on compute capability 9.0 (``cake_cute`` is accepted as an alias).
Supported scope: BF16 HND ``q/k/v`` with shape ``(H, S, 128)``, equal query/KV
head counts, noncausal 64-token query and KV blocks, 1..64 selected KV blocks
per query block (ragged counts allowed), custom finite softmax scales
(including 0 and negative), no LSE / PDL / positional encoding / logits soft cap.

The host planner, the plan object and the generated kernels live in the
self-contained backend package :mod:`flashinfer.cake_vsa_sm90_backend` (host
plan in ``cake_jit.py``, planner in ``cake_planner.py``, sources under
``csrc/`` with their ``build.json`` manifest); this module is its stable
import path and re-exports the backend's public surface.  Planning builds a
:class:`CakeVsaSm90Plan` through :func:`create_plan` (outside CUDA graph
capture); ``plan.run(q, k, v, out=...)`` issues exactly one kernel launch on
the current stream, so a fixed plan is graph-capturable.
"""

from __future__ import annotations

from .cake_vsa_sm90_backend import (  # noqa: F401
    ARCH_BY_CAPABILITY,
    ARCHES,
    BLOCK,
    CLUSTER_VARIANTS,
    HEAD_DIM,
    MAX_NSPLIT,
    MAX_SELECTED,
    MODULES,
    PLAN_HALFWORDS,
    PLAN_META_SPLIT,
    PLAN_META_UNSPLIT,
    POOL_MIN_TILES,
    POOL_RAGGED_MIN_TILES,
    POOL_ROUNDS,
    SMALL_ITEM_ELEMS,
    SMALL_KMAX_VARIANTS,
    SMALL_OCCUPANCY,
    SMALL_STATS_FLOATS,
    SPLIT_CLUSTER_VARIANTS,
    SPLIT_KMAX_VARIANTS,
    STAGES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    CakeVsaSm90Plan,
    arch_for_capability,
    cluster_capacity,
    cluster_cost,
    cluster_variant_for,
    create_plan,
    is_cake_vsa_sm90_supported,
    persistent_stage,
    plan_small,
    plan_vsa_sm90,
    small_kmax_for,
    small_route,
    small_stage,
    split_cluster_variant_for,
    split_cost,
    split_kmax,
)

__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeVsaSm90Plan",
    "cluster_capacity",
    "cluster_variant_for",
    "create_plan",
    "is_cake_vsa_sm90_supported",
    "plan_small",
    "plan_vsa_sm90",
    "small_route",
    "split_cluster_variant_for",
    "split_kmax",
]
