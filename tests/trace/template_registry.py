# Copyright (c) 2025 by FlashInfer team.
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

"""Deterministic discovery of APIs registered with trace templates.

Pytest first collects the complete suite to build the sharding manifest, then
collects one source file per worker process.  A source file must therefore
produce the same parametrized node IDs regardless of modules imported by other
tests.  Import every registration module here, filter out unrelated registry
state, and sort the result by stable API identity.

Keep ``_TRACE_REGISTRATION_MODULES`` synchronized with modules containing
``@flashinfer_api(trace=...)``.  ``test_template_registry.py`` enforces this
inventory.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

TraceRegistryEntry = tuple[Callable[..., Any], Any, str]


_TRACE_REGISTRATION_MODULES = (
    "flashinfer.activation",
    "flashinfer.attention._core",
    "flashinfer.cascade",
    "flashinfer.comm.allreduce",
    "flashinfer.comm.dcp_alltoall",
    "flashinfer.concat_ops",
    "flashinfer.cudnn.decode",
    "flashinfer.cudnn.prefill",
    "flashinfer.cute_dsl.add_rmsnorm_fp4quant",
    "flashinfer.cute_dsl.attention.wrappers.batch_mla",
    "flashinfer.cute_dsl.attention.wrappers.batch_prefill",
    "flashinfer.cute_dsl.rmsnorm_fp4quant",
    "flashinfer.decode",
    "flashinfer.fused_moe.core",
    "flashinfer.fused_moe.cute_dsl.b12x_moe",
    "flashinfer.fused_moe.cute_dsl.fused_moe",
    "flashinfer.fused_moe.fused_routing_dsv3",
    "flashinfer.fused_moe.hash_topk",
    "flashinfer.fused_moe.monomoe",
    "flashinfer.fused_moe.prepare",
    "flashinfer.gdn_decode",
    "flashinfer.gdn_prefill",
    "flashinfer.gemm.gemm_base",
    "flashinfer.gemm.gemm_bf16_fp4",
    "flashinfer.gemm.gemm_svdquant",
    "flashinfer.gemm.kernels.grouped_gemm_masked_blackwell",
    "flashinfer.gemm.routergemm",
    "flashinfer.kda_decode",
    "flashinfer.mamba.selective_state_update",
    "flashinfer.mhc",
    "flashinfer.mla._core",
    "flashinfer.msa_ops.proxy_score",
    "flashinfer.msa_ops.sparse_decode",
    "flashinfer.msa_ops.sparse_prefill",
    "flashinfer.norm",
    "flashinfer.nvfp4_attention_sm120",
    "flashinfer.page",
    "flashinfer.pod",
    "flashinfer.prefill",
    "flashinfer.quantization.fp4_quantization",
    "flashinfer.quantization.fp8_quantization",
    "flashinfer.rope",
    "flashinfer.sampling",
    "flashinfer.sparse",
    "flashinfer.topk",
    "flashinfer.xqa",
)


def trace_registry_entry_key(entry: TraceRegistryEntry) -> tuple[str, str, str]:
    func, _, label = entry
    return func.__module__, func.__qualname__, label


def collect_registered_trace_templates() -> list[TraceRegistryEntry]:
    """Return available trace registrations in deterministic order.

    Optional backends are allowed to be unavailable.  An entry is included
    only when its defining module imports successfully in this process, so a
    partially imported optional module cannot leak state from earlier pytest
    collection.
    """

    available_modules: set[str] = set()
    for module_name in _TRACE_REGISTRATION_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            continue
        available_modules.add(module_name)

    from flashinfer.api_logging import _TRACE_REGISTRY

    entries = [
        entry for entry in _TRACE_REGISTRY if entry[0].__module__ in available_modules
    ]
    keys = [trace_registry_entry_key(entry) for entry in entries]
    if len(keys) != len(set(keys)):
        raise RuntimeError(
            "duplicate trace-template registration identities prevent "
            "deterministic pytest parameter IDs"
        )
    return sorted(entries, key=trace_registry_entry_key)
