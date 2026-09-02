# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host contracts for the SM90 MXFP4 split K1/K2 kernel pair.

The fakes model only workspace metadata and entrypoint signatures.  No CUDA
kernel is launched; this tests that production construction fails closed when
the two concurrent roles cannot share an exact ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
    SplitMegaConfigurationError,
    SplitMegaExecutorRequired,
    SplitMegaMxfp4KernelPair,
    SplitMegaPlan,
    SplitMegaWorkspaceContract,
    SplitMegaWorkspaceMismatch,
    build_mxfp4_split_kernel_pair,
)


@dataclass(frozen=True)
class _Impl:
    mma_tiler_mnk: tuple[int, int, int] = (128, 64, 128)
    cluster_shape_mnk: tuple[int, int, int] = (1, 1, 1)
    use_2cta_instrs: bool = False
    force_static_sched: bool = True
    num_sched_stages: int = 2
    load_balance_mode: str = "static"
    group_hint: int | None = None
    in_kernel_fc2_reduce: bool = False
    token_back_mode: str = "epi_warps"
    clc_bundle_size: int | None = None
    epi_flag_batch: tuple[int, int] = (2, 4)
    flag_batch: int = 1

    @property
    def token_back_by_dispatch(self) -> bool:
        return self.token_back_mode == "reuse_dispatch_warps"


def _impl(
    tile=(128, 64, 128),
    *,
    cluster=(1, 1, 1),
    group_hint=None,
    stages=2,
    load_balance_mode="static",
    force_static_sched=True,
    in_kernel_fc2_reduce=False,
    token_back_mode="epi_warps",
) -> _Impl:
    return _Impl(
        mma_tiler_mnk=tile,
        cluster_shape_mnk=cluster,
        group_hint=group_hint,
        num_sched_stages=stages,
        load_balance_mode=load_balance_mode,
        force_static_sched=force_static_sched,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        token_back_mode=token_back_mode,
    )


class BFloat16:
    pass


class Uint8:
    pass


class Int32:
    pass


@dataclass
class _Spec:
    name: str
    cute_dtype: object
    shape: tuple
    align: int
    nbytes: int
    stride_row_major: tuple


class _FakeKernel:
    instances: list["_FakeKernel"] = []
    mismatch_k2 = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.split_role = kwargs["split_role"]
        self.combine_format = SimpleNamespace(name="bf16")
        handoff_bytes = 80 if self.split_role == "k2" and self.mismatch_k2 else 64
        self.counter_epoch_banks = kwargs.get("split_counter_epoch_banks", 1)
        self.counter_epoch_bank = kwargs.get("split_counter_epoch_bank", 0)

        local_names = [
            "counter" if bank == 0 else f"counter__bank{bank}"
            for bank in range(self.counter_epoch_banks)
        ]
        local_counters = [
            _Spec(name, Uint8, (16,), 16, 16, (1,)) for name in local_names
        ]
        local_data_offset = 16 * self.counter_epoch_banks
        self._local_region_specs = local_counters + [
            _Spec("fc1_output", Uint8, (64,), 16, handoff_bytes, (1,)),
        ]
        self._local_offsets = {name: bank * 16 for bank, name in enumerate(local_names)}
        self._local_offsets["fc1_output"] = local_data_offset
        self._local_total = local_data_offset + handoff_bytes

        join_names = [
            ("split_k2_join_count" if bank == 0 else f"split_k2_join_count__bank{bank}")
            for bank in range(self.counter_epoch_banks)
        ]
        shared_counters = [_Spec(name, Int32, (1,), 16, 4, (1,)) for name in join_names]
        shared_data_offset = 16 * self.counter_epoch_banks
        self._shared_region_specs = shared_counters + [
            _Spec(
                "combine_quant",
                BFloat16,
                (2, 3, 4),
                16,
                48,
                (12, 4, 1),
            ),
        ]
        self._shared_offsets = {name: bank * 16 for bank, name in enumerate(join_names)}
        self._shared_offsets["combine_quant"] = shared_data_offset
        self._shared_total = shared_data_offset + 48
        self.local_zero_i32_count = local_data_offset // 4
        self.shared_zero_i32_count = shared_data_offset // 4
        self.local_counter_bank_spans = tuple(
            (bank * 16, 16) for bank in range(self.counter_epoch_banks)
        )
        self.shared_counter_bank_spans = tuple(
            (bank * 16, 16) for bank in range(self.counter_epoch_banks)
        )
        type(self).instances.append(self)

    def get_workspace_sizes(self):
        return self._local_total, self._shared_total

    def split_k1_entry(
        self,
        activation,
        activation_sf,
        topk_idx,
        topk_weights,
        fc1_weight,
        fc1_weight_sf,
        fc1_weight_dequant_scale,
        local_workspace,
        shared_workspace,
        peer_rank_ptr_mapper_host,
        max_active_clusters,
        stream,
    ):
        raise AssertionError("host contract must not launch K1")

    def split_k2_entry(
        self,
        fc2_weight,
        fc2_weight_sf,
        fc2_weight_dequant_scale,
        local_workspace,
        shared_workspace,
        peer_rank_ptr_mapper_host,
        max_active_clusters,
        stream,
    ):
        raise AssertionError("host contract must not launch K2")


def _problem():
    return SimpleNamespace(
        world_size=4,
        num_tokens_per_rank=1024,
        num_topk=6,
        num_experts_per_rank=96,
        intermediate=6144,
        hidden=7168,
        gate_up_clamp=10.0,
    )


def _build(plan=None, **kwargs):
    return build_mxfp4_split_kernel_pair(
        _problem(),
        plan or SplitMegaPlan(_impl(), _impl(), 80, 52),
        rank=0,
        kernel_class=_FakeKernel,
        ab_dtype=object(),
        sf_padding_block=128,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _reset_fake_kernel():
    _FakeKernel.instances.clear()
    _FakeKernel.mismatch_k2 = False
    yield
    _FakeKernel.instances.clear()
    _FakeKernel.mismatch_k2 = False


def test_plan_allows_independent_m_k_group_and_scheduler_tactics() -> None:
    fc1 = _impl((256, 64, 128), group_hint=80, stages=1)
    fc2 = _impl((128, 64, 256), group_hint=156, stages=3)
    plan = SplitMegaPlan(fc1, fc2, k1_sm_count=80, k2_sm_count=52)

    assert plan.token_padding_block == 64
    assert plan.max_active_clusters_for("k1") == 80
    assert plan.max_active_clusters_for("k2") == 52
    assert plan.impl_for("k1") is fc1
    assert plan.impl_for("k2") is fc2


@pytest.mark.parametrize(
    "fc1_n,fc2_n,expected_handoff,expected_counter",
    [(128, 64, 128, 64), (32, 128, 128, 32), (64, 64, None, 64)],
)
def test_plan_derives_tactic_independent_handoff_and_counter_tiles(
    fc1_n: int,
    fc2_n: int,
    expected_handoff: int | None,
    expected_counter: int,
) -> None:
    plan = SplitMegaPlan(
        _impl((128, fc1_n, 128)),
        _impl((128, fc2_n, 128)),
        80,
        52,
    )
    assert plan.handoff_token_n == expected_handoff
    assert plan.token_padding_block == max(fc1_n, fc2_n)
    assert plan.workspace_counter_tile_tokens == expected_counter


@pytest.mark.parametrize(
    "fc1,fc2,match",
    [
        (_impl(cluster=(1, 1, 1)), _impl(cluster=(1, 2, 1)), "same cluster"),
        (
            _impl(load_balance_mode="atomic_counter"),
            _impl(load_balance_mode="atomic_counter"),
            "load_balance_mode",
        ),
        (_impl(), _impl(in_kernel_fc2_reduce=True), "standalone K3"),
        (
            _impl(),
            _impl(token_back_mode="reuse_dispatch_warps"),
            "direct epilogue combine",
        ),
        (_impl(), _impl(force_static_sched=False), "force_static_sched"),
    ],
)
def test_plan_rejects_semantics_incompatible_with_concurrent_split(
    fc1, fc2, match: str
) -> None:
    with pytest.raises(SplitMegaConfigurationError, match=match):
        SplitMegaPlan(fc1, fc2, 80, 52)


def test_factory_builds_two_roles_with_independent_tactics_and_sm_hints() -> None:
    fc1 = _impl((256, 64, 128), group_hint=240, stages=1)
    fc2 = _impl((128, 64, 256), group_hint=None, stages=3)
    pair = _build(SplitMegaPlan(fc1, fc2, 80, 52))

    assert isinstance(pair, SplitMegaMxfp4KernelPair)
    assert pair.get_workspace_sizes() == (80, 64)
    k1, k2 = _FakeKernel.instances
    assert k1.kwargs["split_role"] == "k1"
    assert k2.kwargs["split_role"] == "k2"
    assert k1.kwargs["static_expert_shape"] == (96, 6144, 7168)
    assert k2.kwargs["static_expert_shape"] == (96, 6144, 7168)
    assert k1.kwargs["mma_tiler_mnk"] == (256, 64, 128)
    assert k2.kwargs["mma_tiler_mnk"] == (128, 64, 256)
    assert k1.kwargs["group_hint"] == 240
    assert k2.kwargs["group_hint"] == 3 * 52
    assert k1.kwargs["split_fc1_tile_m"] is None
    assert k2.kwargs["split_fc1_tile_m"] == 256
    assert not k1.kwargs["fc2_in_kernel_topk_reduce"]
    assert k2.kwargs["token_back_mode"] == "epi_warps"
    assert "token_back_by_dispatch" not in k2.kwargs


def test_independent_token_n_is_carried_into_both_roles() -> None:
    plan = SplitMegaPlan(
        _impl((256, 128, 256)),
        _impl((128, 64, 128)),
        80,
        52,
    )
    _build(plan)
    for kernel in _FakeKernel.instances:
        assert kernel.kwargs["token_padding_block"] == 128
        assert kernel.kwargs["split_handoff_token_n"] == 128
        assert kernel.kwargs["split_fc1_token_n"] == 128
        assert kernel.kwargs["split_workspace_counter_tile_tokens"] == 64


def test_dual_bank_layout_selects_only_the_static_epoch_bank() -> None:
    pair0 = _build(counter_epoch_banks=2, counter_epoch_bank=0)
    pair1 = _build(counter_epoch_banks=2, counter_epoch_bank=1)

    assert pair0.workspace == pair1.workspace
    assert pair0.workspace.counter_epoch_banks == 2
    assert pair0.workspace.local_counter_bank_spans == ((0, 16), (16, 16))
    assert pair0.workspace.shared_counter_bank_spans == ((0, 16), (16, 16))
    assert pair0.selected_counter_bank_span("local") == (0, 16)
    assert pair1.selected_counter_bank_span("local") == (16, 16)
    assert pair1.selected_counter_bank_span("shared") == (16, 16)

    shared = torch.zeros(80, dtype=torch.uint8)
    bank1_join = pair1.join_counter_view(shared)
    assert bank1_join.data_ptr() == shared.data_ptr() + 16
    bank1_join[0] = 9
    assert shared[16:20].view(torch.int32).item() == 9


@pytest.mark.parametrize(
    "banks,bank", [(0, 0), (3, 0), (2, -1), (2, 2), (True, 0), (2, False)]
)
def test_factory_rejects_invalid_counter_bank_selection(banks, bank) -> None:
    with pytest.raises(SplitMegaConfigurationError, match="counter_epoch"):
        _build(counter_epoch_banks=banks, counter_epoch_bank=bank)


def test_workspace_contract_compares_regions_not_only_total_bytes() -> None:
    k1 = _FakeKernel(split_role="k1")
    k2 = _FakeKernel(split_role="k2")
    k2._local_region_specs[1].shape = (32, 2)
    with pytest.raises(SplitMegaWorkspaceMismatch, match="local_regions"):
        SplitMegaWorkspaceContract.require_compatible(k1, k2)


def test_factory_fails_closed_when_k1_k2_workspace_tables_disagree() -> None:
    _FakeKernel.mismatch_k2 = True
    with pytest.raises(SplitMegaWorkspaceMismatch, match="local_total_bytes"):
        _build()


def _runtime_arguments():
    names = (
        "activation",
        "activation_sf",
        "topk_idx",
        "topk_weights",
        "fc1_weight",
        "fc1_weight_sf",
        "fc1_weight_dequant_scale",
        "fc2_weight",
        "fc2_weight_sf",
        "fc2_weight_dequant_scale",
        "local_workspace",
        "shared_workspace",
        "peer_rank_ptr_mapper_host",
    )
    return {name: object() for name in names}


def test_compile_requests_have_narrow_role_abis_and_distinct_streams() -> None:
    pair = _build()
    stream1, stream2 = object(), object()
    k1, k2 = pair.compile_requests(
        _runtime_arguments(),
        k1_stream=stream1,
        k2_stream=stream2,
        options="iket",
    )

    assert k1.role == "k1" and k2.role == "k2"
    assert k1.max_active_clusters == 80
    assert k2.max_active_clusters == 52
    assert k1.kwargs["stream"] is stream1
    assert k2.kwargs["stream"] is stream2
    assert "fc2_weight" not in k1.kwargs
    assert "activation" not in k2.kwargs
    assert "fc1_weight" not in k2.kwargs
    assert "output_activation" not in k1.kwargs
    assert "output_activation" not in k2.kwargs

    with pytest.raises(SplitMegaConfigurationError, match="distinct"):
        pair.compile_requests(
            _runtime_arguments(), k1_stream=stream1, k2_stream=stream1
        )


def test_compile_request_rejects_missing_or_broad_role_entrypoint() -> None:
    pair = _build()
    runtime = _runtime_arguments()
    runtime.pop("local_workspace")
    with pytest.raises(SplitMegaConfigurationError, match="local_workspace"):
        pair.compile_requests(runtime, k1_stream=object(), k2_stream=object())

    pair.k1_kernel.split_k1_entry = lambda **kwargs: None
    with pytest.raises(SplitMegaConfigurationError, match="ABI mismatch"):
        pair.compile_requests(
            _runtime_arguments(), k1_stream=object(), k2_stream=object()
        )


def test_combine_and_join_views_are_exact_peer_visible_regions() -> None:
    pair = _build()
    workspace = torch.zeros(64, dtype=torch.uint8)
    combine = pair.combine_quant_view(workspace)
    join = pair.join_counter_view(workspace)

    assert combine.shape == (2, 3, 4)
    assert combine.dtype is torch.bfloat16
    assert combine.data_ptr() == workspace.data_ptr() + 16
    assert join.shape == (1,)
    assert join.dtype is torch.int32
    assert join.data_ptr() == workspace.data_ptr()


def test_launch_requires_verified_graph_executor_without_sequential_fallback() -> None:
    pair = _build()
    k1, k2 = pair.compile_requests(
        _runtime_arguments(), k1_stream=object(), k2_stream=object()
    )
    with pytest.raises(SplitMegaExecutorRequired, match="sequential"):
        pair.launch(None, k1, k2, k3=object())

    class _Executor:
        def launch(self, got_pair, got_k1, got_k2, *, k3):
            assert (got_pair, got_k1, got_k2) == (pair, k1, k2)
            return k3

    sentinel = object()
    assert pair.launch(_Executor(), k1, k2, k3=sentinel) is sentinel
