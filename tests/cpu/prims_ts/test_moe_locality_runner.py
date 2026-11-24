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

from types import SimpleNamespace

import pytest
import torch

from flashinfer.prims_ts.moe.config_mapper import (
    PrimsTsConfigSpec,
    PrimsTsGemmPair,
    PrimsTsMoeGemmConfig,
    with_locality_overrides,
)
from flashinfer.prims_ts.moe.partition import MoeLocalizedKernelResources
from flashinfer.prims_ts.moe.runner import (
    _PartitionedPrimsTsRunnerMixin,
    _PrimsTsMoERunnerMixin,
    _locality_counter_views,
    _partition_stream_topology,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _compile_arg_tuple


def _pair() -> PrimsTsGemmPair:
    gemm = PrimsTsMoeGemmConfig(
        cfg=PrimsTsConfigSpec(kwargs={"locality_split": 0, "use_clc_fast_drain": 1}),
        prims_ts_gemm_config_index=3,
    )
    return PrimsTsGemmPair(tile_n=16, moe_config_index=2, fc1=gemm, fc2=gemm)


@pytest.mark.cpu_only
def test_locality_tactic_overrides_split_only_tp_n_tp_k_fc2():
    nn = with_locality_overrides(_pair(), split_fc2_activations=False)
    nk = with_locality_overrides(_pair(), split_fc2_activations=True)

    assert nn.fc1.cfg.kwargs["locality_split"] == 1
    assert nn.fc2.cfg.kwargs["locality_split"] == 1
    assert nk.fc1.cfg.kwargs["locality_split"] == 1
    assert nk.fc2.cfg.kwargs["locality_split"] == 2
    assert nk.fc1.cfg.kwargs["use_clc_fast_drain"] == 0
    assert nk.fc2.cfg.kwargs["use_clc_fast_drain"] == 0


@pytest.mark.cpu_only
def test_single_kernel_topology_contains_domain_counts():
    resources = object.__new__(MoeLocalizedKernelResources)
    resources.domain_sm_counts = (101, 97)
    assert _partition_stream_topology(resources) == (
        ("single_kernel", 198, (101, 97), 1),
    )


@pytest.mark.cpu_only
def test_routing_receives_locality_counter_buffers_only_for_localized_handle():
    runner = object.__new__(_FilterRunner)
    resources = object.__new__(MoeLocalizedKernelResources)
    resources.counters = (torch.empty(64), torch.empty(64))

    assert (
        runner._routing_locality_counters({"partition_resources": resources})
        == resources.counters
    )
    assert runner._routing_locality_counters({}) == (None, None)
    assert runner._routing_locality_counters({"partition_resources": None}) == (
        None,
        None,
    )
    multi_stream = SimpleNamespace(counters=resources.counters)
    assert runner._routing_locality_counters({"partition_resources": multi_stream}) == (
        None,
        None,
    )


@pytest.mark.cpu_only
def test_locality_counter_views_use_separate_fc1_and_fc2_lines():
    resources = object.__new__(MoeLocalizedKernelResources)
    resources.counters = (
        torch.empty(64, dtype=torch.int32),
        torch.empty(64, dtype=torch.int32),
    )

    fc1 = _locality_counter_views(resources, 0)
    fc2 = _locality_counter_views(resources, 32)

    for side in range(2):
        assert fc1[side].data_ptr() == resources.counters[side].data_ptr()
        assert fc2[side].data_ptr() - fc1[side].data_ptr() == 128


@pytest.mark.cpu_only
def test_non_locality_compile_args_leave_side1_and_counters_absent():
    marker = object()
    io = {
        name: marker
        for name in (
            "a_dp",
            "b_dp",
            "sfa_dp",
            "sfb_dp",
            "c0_dp",
            "sf_c_dp",
            "tile_idx_dp",
            "route_map_dp",
            "mn_limit_dp",
            "num_non_exiting_ctas_dp",
            "act_dp",
            "per_token_sf_a_dp",
            "per_token_sf_b_dp",
            "bias_dp",
            "scale_c_dp",
            "scale_gate_dp",
            "gemm1_alpha_dp",
            "gemm1_beta_dp",
            "gemm1_clamp_limit_dp",
        )
    }
    io.update(
        shape=(1, 2, 3, 4, 5),
        launch_early_exit_max_token_ctas=1,
        cfg=SimpleNamespace(
            is_swap_ab=False, has_gated_epilogue=False, fused_finalize_top_k=0
        ),
    )

    args = _compile_arg_tuple(io, marker)

    assert args[27:33] == (None,) * 6
    assert args[33] == 0
    assert args[34:36] == (None, None)


class _FilterRunner(_PartitionedPrimsTsRunnerMixin, _PrimsTsMoERunnerMixin):
    def __init__(self, topology):
        self._partition_strategy = "tp_n_tp_n"
        self._partition_topology = topology
        self._localized = bool(topology and topology[0][0] == "single_kernel")
        self._cache_key_static_extras = ()
        self.intermediate_size = 256
        self.hidden_size = 512

    def _validate_partition_mode(self):
        return None

    def _check_cga_tactic(self, _fc1, _fc2):
        return None

    def _check_partition_tactic(self, _fc1, _fc2):
        return None

    def _map_tactic(self, tactic, _moe_inputs, kwargs=None):
        del kwargs
        persistent = tactic == "persistent"
        cfg = SimpleNamespace(
            is_persistent=persistent,
            fallback_cluster_shape=None,
        )
        spec = SimpleNamespace(build=lambda: cfg)
        gemm = SimpleNamespace(cfg=spec)
        return SimpleNamespace(fc1=gemm, fc2=gemm)


@pytest.mark.cpu_only
def test_geometry_filter_cache_separates_stream_and_single_kernel_topologies():
    _PrimsTsMoERunnerMixin._geometry_tactics_cache.clear()
    inputs = [torch.empty(1) for _ in range(8)]
    localized = _FilterRunner((("single_kernel", 198, (101, 97), 1),))
    streams = _FilterRunner((("green", 99, 0, 0, 0),))

    assert localized._filter_partition_geometry_tactics(
        ["static", "persistent"], inputs, ("same-shape",)
    ) == ["persistent"]
    assert streams._filter_partition_geometry_tactics(
        ["static", "persistent"], inputs, ("same-shape",)
    ) == ["static", "persistent"]


@pytest.mark.cpu_only
def test_explicit_static_tactic_is_rejected():
    runner = _FilterRunner((("single_kernel", 198, (101, 97), 1),))
    static_cfg = SimpleNamespace(
        is_persistent=False,
        fallback_cluster_shape=None,
    )

    with pytest.raises(ValueError, match="requires persistent tactics"):
        runner._check_localized_tactic(static_cfg, static_cfg)


@pytest.mark.cpu_only
def test_partition_resource_type_must_match_runner_topology():
    runner = _FilterRunner((("single_kernel", 198, (101, 97), 1),))
    runner._partition_shards = lambda _kwargs: (object(), (), (), None, None)

    with pytest.raises(ValueError, match="does not match"):
        runner._run_partitioned(
            builder=None,
            fc1_cfg=None,
            fc2_cfg=None,
            common_io_kwargs={},
            fc1_label="fc1",
            fc2_label="fc2",
            gemm1_output=None,
            gemm2_output=None,
            output=None,
            expert_weights=None,
            expanded_idx_to_permuted_idx=None,
            total_num_padded_tokens=None,
            num_tokens=0,
            kwargs={},
        )
