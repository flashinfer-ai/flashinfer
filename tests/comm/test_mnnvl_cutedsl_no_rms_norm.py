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

"""Contracts for the MNNVL CuTe DSL backend with the fused RMSNorm disabled.

``apply_rms_norm=False`` compiles the collective without its normalisation, so
the chain stops after finalize + shared-expert add + all-reduce (+ residual)
and materialises that value into ``residual_out``.

These run at four or eight ranks against hidden 5120, unlike
test_mnnvl_cutedsl_numerical_contract.py which is pinned to hidden 8192 and
eight or sixteen ranks.
"""

import os

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion
from flashinfer.comm.mnnvl_cutedsl import (
    BT_ONLY_CONFIG,
    DEFAULT_CONFIG,
    HT_ONLY_CONFIG,
    LL_ONLY_CONFIG,
    NO_NORM_CONFIG,
    KernelTarget,
    MNNVLCuteDSLConfig,
    MRangeDispatch,
    ProtocolKind,
    StaticProfile,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_bt import (
    BTAllReduceTuning,
    BTCollectiveTuning,
    BTFinalizeTuning,
)
from flashinfer.comm.mnnvl_cutedsl_ar import MNNVLCuteDSLAllReduceFusionWorkspace
from flashinfer.utils import is_sm100a_supported

HIDDEN_SIZE = 5120
TOP_K = 6
RMS_EPS = 1e-6
PROTOCOL_CONFIGS = {
    "ll": LL_ONLY_CONFIG,
    "bt": BT_ONLY_CONFIG,
    "ht": HT_ONLY_CONFIG,
}
pytestmark = [pytest.mark.gpu_4, pytest.mark.arch_blackwell]


@pytest.fixture(scope="module")
def distributed_group():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (4, 8):
        pytest.skip("Run this test with four or eight distributed ranks")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not is_sm100a_supported(device):
        pytest.skip("SM100 or newer data-center Blackwell is required")
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", device_id=device)
    try:
        yield dist.group.WORLD
    finally:
        if owns_group:
            dist.destroy_process_group()


def _workspace(protocol: str, capacity_m: int, group, **overrides):
    kwargs = {
        "tp_size": dist.get_world_size(group),
        "tp_rank": dist.get_rank(group),
        "max_token_num": capacity_m,
        "hidden_dim": HIDDEN_SIZE,
        "dtype": torch.bfloat16,
        "group": group,
        "top_k": TOP_K,
        "rms_eps": RMS_EPS,
        "config": PROTOCOL_CONFIGS[protocol],
    }
    kwargs.update(overrides)
    workspace = MNNVLCuteDSLAllReduceFusionWorkspace(**kwargs)
    torch.cuda.synchronize()
    dist.barrier(group)
    return workspace


def _sanitize_negative_zero(value: torch.Tensor) -> torch.Tensor:
    value = value.clone()
    bits = value.view(torch.int16)
    bits.masked_fill_(bits == -32768, 0)
    return value


def _ordered_reduce(local: torch.Tensor, group) -> torch.Tensor:
    peers = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, local, group=group)
    reduced = torch.zeros_like(local, dtype=torch.float32)
    for peer in peers:
        reduced.add_(peer.float())
    return reduced


def _local_finalize(routed, weights, indices, shared) -> torch.Tensor:
    m = weights.shape[0]
    local = torch.zeros((m, HIDDEN_SIZE), dtype=torch.float32, device="cuda")
    for route in range(TOP_K):
        rows = indices[:, route].to(torch.int64)
        torch.addcmul(
            local,
            routed.index_select(0, rows).float(),
            weights[:, route, None].float(),
            out=local,
        )
    local.add_(shared.float())
    return _sanitize_negative_zero(local.to(torch.bfloat16))


@pytest.mark.parametrize("protocol", ("ll", "bt"))
@torch.inference_mode()
def test_disabled_norm_materializes_all_reduced_finalize(distributed_group, protocol):
    """residual_out carries finalize + shared expert + all-reduce, unnormalised."""
    group = distributed_group
    rank = dist.get_rank(group)
    m = dist.get_world_size(group)
    workspace = _workspace(protocol, m, group, add_residual=False, apply_rms_norm=False)
    try:
        generator = torch.Generator(device="cuda").manual_seed(4100 + rank)
        routed = torch.randn(
            m * TOP_K,
            HIDDEN_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        weights = torch.randn(
            m, TOP_K, generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        shared = torch.randn(
            m, HIDDEN_SIZE, generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.arange(m * TOP_K, dtype=torch.int32, device="cuda").reshape(
            m, TOP_K
        )
        residual_out = torch.empty(m, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        allreduce_fusion(
            input=routed,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
            launch_with_pdl=True,
            residual_out=residual_out,
            expanded_idx_to_permuted_idx=indices,
            expert_scale_factor=weights,
            shared_expert_output=shared,
        )
        expected = _sanitize_negative_zero(
            _ordered_reduce(
                _local_finalize(routed, weights, indices, shared), group
            ).to(torch.bfloat16)
        )
        assert torch.equal(residual_out.view(torch.int16), expected.view(torch.int16))
    finally:
        workspace.destroy()


@pytest.mark.parametrize("protocol", ("ll", "bt"))
@torch.inference_mode()
def test_disabled_norm_returns_reduce_when_output_not_preallocated(
    distributed_group, protocol
):
    """The returned tensor holds the reduce even when residual_out is omitted.

    With the norm compiled out the kernel writes only the residual slot. If the
    wrapper lets the protocol allocate an unrelated norm buffer and returns
    that, the caller gets uninitialised memory instead of the reduced value.
    """
    group = distributed_group
    rank = dist.get_rank(group)
    m = dist.get_world_size(group)
    workspace = _workspace(protocol, m, group, add_residual=False, apply_rms_norm=False)
    try:
        generator = torch.Generator(device="cuda").manual_seed(4700 + rank)
        routed = torch.randn(
            m * TOP_K,
            HIDDEN_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        weights = torch.randn(
            m, TOP_K, generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        shared = torch.randn(
            m, HIDDEN_SIZE, generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.arange(m * TOP_K, dtype=torch.int32, device="cuda").reshape(
            m, TOP_K
        )
        returned = allreduce_fusion(
            input=routed,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
            launch_with_pdl=True,
            expanded_idx_to_permuted_idx=indices,
            expert_scale_factor=weights,
            shared_expert_output=shared,
        )
        expected = _sanitize_negative_zero(
            _ordered_reduce(
                _local_finalize(routed, weights, indices, shared), group
            ).to(torch.bfloat16)
        )
        assert torch.equal(returned.view(torch.int16), expected.view(torch.int16))
    finally:
        workspace.destroy()


@torch.inference_mode()
def test_disabled_norm_rejects_norm_operands(distributed_group):
    """rms_gamma and norm_out are meaningless once the norm is compiled out."""
    group = distributed_group
    m = dist.get_world_size(group)
    workspace = _workspace("bt", m, group, add_residual=False, apply_rms_norm=False)
    try:
        local = torch.randn(m, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        residual_out = torch.empty_like(local)
        gamma = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="rms_gamma"):
            allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                residual_out=residual_out,
                rms_gamma=gamma,
            )
        with pytest.raises(ValueError, match="norm_out"):
            allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                residual_out=residual_out,
                norm_out=torch.empty_like(local),
            )
    finally:
        workspace.destroy()


@torch.inference_mode()
def test_disabled_norm_requires_a_residual_output(distributed_group):
    """Disabling both the norm and residual_out would leave no output at all."""
    group = distributed_group
    m = dist.get_world_size(group)
    with pytest.raises(ValueError, match="write_residual_output"):
        _workspace(
            "bt",
            m,
            group,
            apply_rms_norm=False,
            write_residual_output=False,
        )


@pytest.mark.parametrize("protocol", ("ll", "bt"))
@torch.inference_mode()
def test_disabled_norm_plain_all_reduce(distributed_group, protocol):
    """Without the norm and the residual, the all-reduce path is a plain all-reduce.

    The token count is not a multiple of the TP size.
    """
    group = distributed_group
    rank = dist.get_rank(group)
    m = 1025
    workspace = _workspace(protocol, m, group, add_residual=False, apply_rms_norm=False)
    try:
        generator = torch.Generator(device="cuda").manual_seed(5300 + rank)
        local = _sanitize_negative_zero(
            torch.randn(
                m, HIDDEN_SIZE, generator=generator, dtype=torch.bfloat16, device="cuda"
            )
        )
        expected = _sanitize_negative_zero(
            _ordered_reduce(local, group).to(torch.bfloat16)
        )
        for _ in range(3):
            residual_out = torch.empty_like(local)
            returned = allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=True,
                residual_out=residual_out,
            )
            assert returned.data_ptr() == residual_out.data_ptr()
            assert torch.equal(
                residual_out.view(torch.int16), expected.view(torch.int16)
            )
    finally:
        workspace.destroy()


@torch.inference_mode()
def test_high_throughput_rejects_disabled_norm(distributed_group):
    """HT fuses the norm into a warp group, so it cannot be compiled out."""
    group = distributed_group
    tp_size = dist.get_world_size(group)
    if not any(
        profile.tp_size == tp_size
        and profile.hidden_size == HIDDEN_SIZE
        and profile.top_k == TOP_K
        for profile in HT_ONLY_CONFIG.profiles
    ):
        # HT is structurally unreachable at hidden 5120 for tp >= 8, so there is
        # no HT profile to reject the flag; the workspace fails earlier and for
        # a different reason. See the note in kernel_ht/protocol.py.
        pytest.skip(f"no HT profile at tp_size={tp_size}, hidden={HIDDEN_SIZE}")
    with pytest.raises(NotImplementedError, match="apply_rms_norm"):
        _workspace("ht", tp_size, group, add_residual=False, apply_rms_norm=False)


@torch.inference_mode()
def test_no_norm_config_routes_without_a_norm_tail(distributed_group):
    """NO_NORM_CONFIG covers this shape and never routes to HT.

    HT cannot compile the norm out, so a norm-free config has no HT tail; BT
    carries the range to the top.
    """
    group = distributed_group
    tp_size = dist.get_world_size(group)
    profile = NO_NORM_CONFIG.resolve(
        tp_size=tp_size,
        hidden_size=HIDDEN_SIZE,
        top_k=TOP_K,
        dtype=torch.bfloat16,
        capacity_m=4096,
    )
    for routes in (profile.finalize_routes, profile.all_reduce_routes):
        protocols = {target.protocol.value for target in routes.targets}
        assert "ht" not in protocols
        assert routes.is_unbounded


@torch.inference_mode()
def test_norm_free_config_rejects_norm_on_workspace(distributed_group):
    """NO_NORM_CONFIG's boundaries are only valid for norm-free kernels."""
    group = distributed_group
    m = dist.get_world_size(group)
    with pytest.raises(ValueError, match="apply_rms_norm"):
        MNNVLCuteDSLAllReduceFusionWorkspace(
            tp_size=m,
            tp_rank=dist.get_rank(group),
            max_token_num=m,
            hidden_dim=HIDDEN_SIZE,
            dtype=torch.bfloat16,
            group=group,
            top_k=TOP_K,
            rms_eps=RMS_EPS,
            apply_rms_norm=True,
            config=NO_NORM_CONFIG,
        )


@torch.inference_mode()
def test_norm_on_config_rejects_norm_free_workspace(distributed_group):
    """DEFAULT_CONFIG's boundaries were measured with the norm enabled."""
    group = distributed_group
    m = dist.get_world_size(group)
    with pytest.raises(ValueError, match="apply_rms_norm"):
        MNNVLCuteDSLAllReduceFusionWorkspace(
            tp_size=m,
            tp_rank=dist.get_rank(group),
            max_token_num=m,
            hidden_dim=HIDDEN_SIZE,
            dtype=torch.bfloat16,
            group=group,
            top_k=TOP_K,
            rms_eps=RMS_EPS,
            add_residual=False,
            apply_rms_norm=False,
            config=DEFAULT_CONFIG,
        )


@torch.inference_mode()
def test_no_norm_config_materializes_all_reduced_finalize(distributed_group):
    """End to end through NO_NORM_CONFIG's own routing."""
    group = distributed_group
    rank = dist.get_rank(group)
    m = dist.get_world_size(group)
    workspace = MNNVLCuteDSLAllReduceFusionWorkspace(
        tp_size=m,
        tp_rank=rank,
        max_token_num=m,
        hidden_dim=HIDDEN_SIZE,
        dtype=torch.bfloat16,
        group=group,
        top_k=TOP_K,
        rms_eps=RMS_EPS,
        add_residual=False,
        apply_rms_norm=False,
        config=NO_NORM_CONFIG,
    )
    torch.cuda.synchronize()
    dist.barrier(group)
    try:
        generator = torch.Generator(device="cuda").manual_seed(4900 + rank)
        routed = torch.randn(
            m * TOP_K,
            HIDDEN_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        weights = torch.randn(
            m, TOP_K, generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        shared = torch.randn(
            m, HIDDEN_SIZE, generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.arange(m * TOP_K, dtype=torch.int32, device="cuda").reshape(
            m, TOP_K
        )
        returned = allreduce_fusion(
            input=routed,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
            launch_with_pdl=True,
            expanded_idx_to_permuted_idx=indices,
            expert_scale_factor=weights,
            shared_expert_output=shared,
        )
        expected = _sanitize_negative_zero(
            _ordered_reduce(
                _local_finalize(routed, weights, indices, shared), group
            ).to(torch.bfloat16)
        )
        assert torch.equal(returned.view(torch.int16), expected.view(torch.int16))
    finally:
        workspace.destroy()


def _fused_bt_config(
    tp_size: int, apply_rms_norm: bool, lag: int
) -> MNNVLCuteDSLConfig:
    def route(preset):
        return MRangeDispatch(
            upper_bounds=(None,), targets=(KernelTarget(ProtocolKind.BT, preset),)
        )

    return MNNVLCuteDSLConfig(
        profiles=(
            StaticProfile(
                tp_size=tp_size,
                hidden_size=HIDDEN_SIZE,
                top_k=TOP_K,
                dtype=torch.bfloat16,
                finalize_routes=route(BTFinalizeTuning()),
                all_reduce_routes=route(
                    BTAllReduceTuning(
                        collective=BTCollectiveTuning(rms_threads=512),
                        fused=True,
                        fused_lag=lag,
                    )
                ),
            ),
        ),
        applies_rms_norm=apply_rms_norm,
    )


@pytest.mark.parametrize("lag", (1, 3))
@pytest.mark.parametrize("m", (3, 1025))
@torch.inference_mode()
def test_fused_bt_plain_all_reduce(distributed_group, lag, m):
    """The one-launch BT all-reduce matches the rank-ordered FP32 sum bitwise.

    It shares its stage counter and mailboxes with the three-launch finalize
    path, so the calls are interleaved on one workspace.
    """
    group = distributed_group
    rank = dist.get_rank(group)
    tp_size = dist.get_world_size(group)
    workspace = _workspace(
        "bt",
        m,
        group,
        add_residual=False,
        apply_rms_norm=False,
        config=_fused_bt_config(tp_size, False, lag),
    )
    try:
        generator = torch.Generator(device="cuda").manual_seed(6100 + rank)
        local = _sanitize_negative_zero(
            torch.randn(
                m, HIDDEN_SIZE, generator=generator, dtype=torch.bfloat16, device="cuda"
            )
        )
        expected = _sanitize_negative_zero(
            _ordered_reduce(local, group).to(torch.bfloat16)
        )
        routed = torch.randn(
            m * TOP_K,
            HIDDEN_SIZE,
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        weights = torch.randn(
            m, TOP_K, generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        shared = torch.randn(
            m, HIDDEN_SIZE, generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.arange(m * TOP_K, dtype=torch.int32, device="cuda").reshape(
            m, TOP_K
        )
        expected_finalize = _sanitize_negative_zero(
            _ordered_reduce(
                _local_finalize(routed, weights, indices, shared), group
            ).to(torch.bfloat16)
        )
        for _ in range(4):
            residual_out = torch.empty_like(local)
            allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=True,
                residual_out=residual_out,
            )
            assert torch.equal(
                residual_out.view(torch.int16), expected.view(torch.int16)
            )
            finalize_out = torch.empty_like(local)
            allreduce_fusion(
                input=routed,
                workspace=workspace,
                pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
                launch_with_pdl=True,
                residual_out=finalize_out,
                expanded_idx_to_permuted_idx=indices,
                expert_scale_factor=weights,
                shared_expert_output=shared,
            )
            assert torch.equal(
                finalize_out.view(torch.int16), expected_finalize.view(torch.int16)
            )
    finally:
        workspace.destroy()


@pytest.mark.parametrize("lag", (1, 3))
@pytest.mark.parametrize("m", (3, 1025))
@torch.inference_mode()
def test_fused_bt_residual_rms_norm(distributed_group, lag, m):
    """With the norm on, the one-launch BT path matches an FP32 reference."""
    group = distributed_group
    rank = dist.get_rank(group)
    tp_size = dist.get_world_size(group)
    workspace = _workspace("bt", m, group, config=_fused_bt_config(tp_size, True, lag))
    try:
        generator = torch.Generator(device="cuda").manual_seed(6300 + rank)
        local = _sanitize_negative_zero(
            torch.randn(
                m, HIDDEN_SIZE, generator=generator, dtype=torch.bfloat16, device="cuda"
            )
        )
        residual = torch.randn(
            m,
            HIDDEN_SIZE,
            generator=torch.Generator(device="cuda").manual_seed(6400),
            dtype=torch.bfloat16,
            device="cuda",
        )
        gamma = torch.randn(
            HIDDEN_SIZE,
            generator=torch.Generator(device="cuda").manual_seed(6500),
            dtype=torch.bfloat16,
            device="cuda",
        )
        prenorm = (_ordered_reduce(local, group) + residual.float()).to(torch.bfloat16)
        values = prenorm.float()
        expected_norm = (
            values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        ) * gamma.float()
        for _ in range(3):
            residual_out = torch.empty_like(local)
            norm_out = torch.empty_like(local)
            allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=True,
                residual_in=residual,
                rms_gamma=gamma,
                rms_eps=RMS_EPS,
                residual_out=residual_out,
                norm_out=norm_out,
            )
            assert torch.equal(
                residual_out.view(torch.int16),
                _sanitize_negative_zero(prenorm).view(torch.int16),
            )
            torch.testing.assert_close(
                norm_out.float(), expected_norm, rtol=2**-7, atol=2**-6
            )
    finally:
        workspace.destroy()


@torch.inference_mode()
def test_fused_bt_dynamic_schedule_across_token_counts(distributed_group):
    """The per-stage work-ticket counters stay consistent when consecutive
    calls use different token counts."""
    group = distributed_group
    rank = dist.get_rank(group)
    tp_size = dist.get_world_size(group)
    capacity = 1025
    workspace = _workspace(
        "bt",
        capacity,
        group,
        add_residual=False,
        apply_rms_norm=False,
        config=_fused_bt_config(tp_size, False, 3),
    )
    try:
        generator = torch.Generator(device="cuda").manual_seed(6700 + rank)
        local = _sanitize_negative_zero(
            torch.randn(
                capacity,
                HIDDEN_SIZE,
                generator=generator,
                dtype=torch.bfloat16,
                device="cuda",
            )
        )
        expected = _sanitize_negative_zero(
            _ordered_reduce(local, group).to(torch.bfloat16)
        )
        for m in (1025, 3, 700, 1, 1025, 512, 2, 1025):
            residual_out = torch.empty_like(local[:m])
            allreduce_fusion(
                input=local[:m],
                workspace=workspace,
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=True,
                residual_out=residual_out,
            )
            assert torch.equal(
                residual_out.view(torch.int16), expected[:m].view(torch.int16)
            )
    finally:
        workspace.destroy()
