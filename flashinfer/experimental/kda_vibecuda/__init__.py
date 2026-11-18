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

"""Experimental VibeCUDA recurrent KDA prefill backend (SM100a / SM103a)."""

from .backend import (
    _FAM2_HEADS_BY_TARGET,
    _FAST_PLAN_ENABLED,
    _FAST_PLAN_LIMIT,
    _FAST_PLAN_TARGETS,
    _FN16_FIXED_LAYOUT_TARGETS,
    _VibeCUDAPrefillState,
    _fast_plan_device_supported,
    _fast_plan_key,
    _get_vibecuda_prefill_module,
    _get_vibecuda_stream_workspace,
    _run_vibecuda_kda_prefill,
    _try_run_fast_vibecuda_kda_prefill,
    _vibecuda_beta_tma_source,
    _vibecuda_descriptor_storage,
    _vibecuda_dummy_state,
    _vibecuda_fn16_buffers,
    _vibecuda_fn16_chain_schedule,
    _vibecuda_fn16_prepare_ctas,
    _vibecuda_kda_prefill_is_eligible,
    _vibecuda_persistent_planner_buffers,
    _vibecuda_pool,
    _vibecuda_prefill_target,
    _vibecuda_sorted_seq_order,
    _vibecuda_split_buffers,
    _vibecuda_split_lookback_flags,
    _vibecuda_split_out_buffer,
    _vibecuda_state,
    _vibecuda_use_transient_pools,
)
from .jit import (
    VibeCUDAPrefillTarget,
    gen_vibecuda_flash_kda_module,
    get_vibecuda_csrc_dir,
    get_vibecuda_flash_kda_uri,
    load_vibecuda_flash_kda_module,
)

__all__ = [
    "VibeCUDAPrefillTarget",
    "_FAM2_HEADS_BY_TARGET",
    "_FAST_PLAN_ENABLED",
    "_FAST_PLAN_LIMIT",
    "_FAST_PLAN_TARGETS",
    "_FN16_FIXED_LAYOUT_TARGETS",
    "_VibeCUDAPrefillState",
    "_fast_plan_device_supported",
    "_fast_plan_key",
    "_get_vibecuda_prefill_module",
    "_get_vibecuda_stream_workspace",
    "_run_vibecuda_kda_prefill",
    "_try_run_fast_vibecuda_kda_prefill",
    "_vibecuda_beta_tma_source",
    "_vibecuda_descriptor_storage",
    "_vibecuda_dummy_state",
    "_vibecuda_fn16_buffers",
    "_vibecuda_fn16_chain_schedule",
    "_vibecuda_fn16_prepare_ctas",
    "_vibecuda_kda_prefill_is_eligible",
    "_vibecuda_persistent_planner_buffers",
    "_vibecuda_pool",
    "_vibecuda_prefill_target",
    "_vibecuda_sorted_seq_order",
    "_vibecuda_split_buffers",
    "_vibecuda_split_lookback_flags",
    "_vibecuda_split_out_buffer",
    "_vibecuda_state",
    "_vibecuda_use_transient_pools",
    "gen_vibecuda_flash_kda_module",
    "get_vibecuda_csrc_dir",
    "get_vibecuda_flash_kda_uri",
    "load_vibecuda_flash_kda_module",
]
