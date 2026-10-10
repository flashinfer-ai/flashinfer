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

"""Protocol numerical contracts for the hidden 6144 MNNVL CuTe DSL profiles.

The same contracts as test_mnnvl_cutedsl_numerical_contract.py (pinned to
hidden 8192 at eight or sixteen ranks), run against hidden 6144, top_k 8 at
four or eight ranks. Unlike hidden 5120, HT is reachable at tp=8 here, so the
HT cases run at both rank counts. The norm-free contracts run here too, since
test_mnnvl_cutedsl_no_rms_norm.py is pinned to hidden 5120.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from flashinfer.comm import (
    AllReduceFusionPattern,
    allreduce_fusion,
)
from flashinfer.comm.mnnvl_cutedsl import (
    BT_ONLY_CONFIG,
    HT_ONLY_CONFIG,
    LL_ONLY_CONFIG,
    NO_NORM_CONFIG,
)
from flashinfer.comm.mnnvl_cutedsl.config import ProtocolKind
from flashinfer.comm.mnnvl_cutedsl_ar import (
    MNNVLCuteDSLAllReduceFusionWorkspace,
)
from flashinfer.utils import is_sm100a_supported


HIDDEN_SIZE = 6144
TOP_K = 8
RMS_EPS = 1e-6
WEIGHT_BIAS = 1.0
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


def _ordered_bf16(bits: torch.Tensor) -> torch.Tensor:
    unsigned = bits.to(torch.int32) & 0xFFFF
    negative = (unsigned & 0x8000) != 0
    return torch.where(negative, 0x8000 - (unsigned & 0x7FFF), 0x8000 + unsigned)


def _max_bf16_ulp(actual: torch.Tensor, reference: torch.Tensor) -> int:
    actual_bits = actual.view(torch.int16)
    reference_bits = reference.to(torch.bfloat16).view(torch.int16)
    ulp = (_ordered_bf16(actual_bits) - _ordered_bf16(reference_bits)).abs()
    both_zero = (actual == 0) & (reference == 0)
    return int(torch.where(both_zero, 0, ulp).max().item())


def _sanitize_negative_zero(value: torch.Tensor) -> torch.Tensor:
    value = value.clone()
    bits = value.view(torch.int16)
    bits.masked_fill_(bits == -32768, 0)
    return value


def _rms_norm(prenorm: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    prenorm_f32 = prenorm.float()
    inv_rms = torch.rsqrt(prenorm_f32.square().mean(dim=-1, keepdim=True) + RMS_EPS)
    return (prenorm_f32 * inv_rms * (gamma.float() + WEIGHT_BIAS)).to(torch.bfloat16)


def _ordered_reduce(local: torch.Tensor, group) -> torch.Tensor:
    peers = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, local, group=group)
    reduced = torch.zeros_like(local, dtype=torch.float32)
    for peer in peers:
        reduced.add_(peer.float())
    return reduced


def _make_symmetric_reference(shape: tuple[int, int], group):
    input_tensor = symm_mem.empty(shape, dtype=torch.bfloat16, device="cuda")
    output_tensor = symm_mem.empty(shape, dtype=torch.bfloat16, device="cuda")
    input_handle = symm_mem.rendezvous(input_tensor, group)
    output_handle = symm_mem.rendezvous(output_tensor, group)
    if not input_handle.multicast_ptr or not output_handle.multicast_ptr:
        pytest.skip("Symmetric NVLS mappings are required")
    return input_tensor, output_tensor, input_handle, output_handle


def _symmetric_reduce(local: torch.Tensor, reference, group) -> torch.Tensor:
    input_tensor, output_tensor, _, _ = reference
    input_tensor.copy_(local)
    torch.ops.symm_mem.multimem_one_shot_all_reduce_out(
        input_tensor,
        "sum",
        group.group_name,
        output_tensor,
    )
    return output_tensor.clone()


def _protocol_prenorm(
    protocol: str,
    local: torch.Tensor,
    residual: torch.Tensor,
    group,
    symmetric_reference,
) -> torch.Tensor:
    if protocol == "ht":
        reduced = _symmetric_reduce(local, symmetric_reference, group).float()
    else:
        reduced = _ordered_reduce(local, group)
    return _sanitize_negative_zero((reduced + residual.float()).to(torch.bfloat16))


def _assert_prenorm_contract(
    protocol: str, actual: torch.Tensor, reference: torch.Tensor
) -> None:
    if protocol in ("ll", "ht"):
        assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
    else:
        assert _max_bf16_ulp(actual, reference) <= 1


def _assert_norm_contract(
    protocol: str,
    actual: torch.Tensor,
    actual_prenorm: torch.Tensor,
    reference_prenorm: torch.Tensor,
    gamma: torch.Tensor,
) -> None:
    assert _max_bf16_ulp(actual, _rms_norm(actual_prenorm, gamma)) <= 1
    end_to_end_limit = 1 if protocol == "ht" else 2
    assert (
        _max_bf16_ulp(actual, _rms_norm(reference_prenorm, gamma)) <= end_to_end_limit
    )


def _local_finalize(
    routed: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor:
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


def _order_sensitive_local(m: int, rank: int, world_size: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(3100 + rank)
    local = torch.randn(
        m,
        HIDDEN_SIZE,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    for column in range(world_size):
        local[:, column] = 0
        if rank == column:
            local[:, column] = 2**24
        elif rank == (column + 1) % world_size:
            local[:, column] = 1
        elif rank == (column + 2) % world_size:
            local[:, column] = -(2**24)
    return local


def _past_bt_split(tp_size: int) -> int:
    """Smallest M past every finite BT_ONLY bound, so both BT presets run.

    Derived from the profile rather than hard-coded, so retuning the splits
    cannot leave a stale constant here that quietly stops exercising PRESET_1.
    """
    (profile,) = (
        p
        for p in BT_ONLY_CONFIG.profiles
        if p.matches(
            tp_size=tp_size,
            hidden_size=HIDDEN_SIZE,
            top_k=TOP_K,
            dtype=torch.bfloat16,
        )
    )
    bounds = [
        bound
        for route in (profile.finalize_routes, profile.all_reduce_routes)
        for bound in route.upper_bounds
        if bound is not None
    ]
    return max(bounds) + 1


def _workspace(protocol: str | None, capacity_m: int, group, **overrides):
    kwargs = {
        "tp_size": dist.get_world_size(group),
        "tp_rank": dist.get_rank(group),
        "max_token_num": capacity_m,
        "hidden_dim": HIDDEN_SIZE,
        "dtype": torch.bfloat16,
        "group": group,
        "top_k": TOP_K,
        "rms_eps": RMS_EPS,
        "weight_bias": WEIGHT_BIAS,
    }
    if protocol is not None:
        kwargs["config"] = PROTOCOL_CONFIGS[protocol]
    kwargs.update(overrides)
    workspace = MNNVLCuteDSLAllReduceFusionWorkspace(**kwargs)
    torch.cuda.synchronize()
    dist.barrier(group)
    return workspace


def _finalize_inputs(m: int, rank: int):
    """Per-rank MoE finalize operands: each rank contributes one token's worth."""
    routed = torch.zeros(m * TOP_K, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    weights = torch.zeros(m, TOP_K, dtype=torch.bfloat16, device="cuda")
    shared = torch.zeros(m, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    rank_generator = torch.Generator(device="cuda").manual_seed(3300 + rank)
    routed[rank * TOP_K : (rank + 1) * TOP_K].normal_(generator=rank_generator)
    weights[rank].normal_(generator=rank_generator)
    shared[rank].normal_(generator=rank_generator)
    indices = torch.arange(m * TOP_K, dtype=torch.int32, device="cuda").reshape(
        m, TOP_K
    )
    return routed, weights, shared, indices


def _check_disabled_norm_pair(protocol, workspace, m, rank, world_size, group):
    """Norm-free AR and finalize land the reduced (+ residual) value in residual_out.

    `protocol` picks the contract: LL is bit-exact on the AR prenorm, BT within
    one ulp; the finalize prenorm is bit-exact for both, as with the norm on.
    """
    residual = torch.randn(
        m,
        HIDDEN_SIZE,
        generator=torch.Generator(device="cuda").manual_seed(4200),
        dtype=torch.bfloat16,
        device="cuda",
    )
    local = _order_sensitive_local(m, rank, world_size)
    residual_out = torch.empty_like(local)
    allreduce_fusion(
        input=local,
        workspace=workspace,
        pattern=AllReduceFusionPattern.kARResidualRMSNorm,
        launch_with_pdl=True,
        residual_in=residual,
        residual_out=residual_out,
        weight_bias=WEIGHT_BIAS,
    )
    _assert_prenorm_contract(
        protocol,
        residual_out,
        _protocol_prenorm(protocol, local, residual, group, None),
    )

    routed, weights, shared, indices = _finalize_inputs(m, rank)
    zero_residual = torch.zeros_like(residual)
    residual_out.zero_()
    allreduce_fusion(
        input=routed,
        workspace=workspace,
        pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
        launch_with_pdl=True,
        residual_in=zero_residual,
        residual_out=residual_out,
        expanded_idx_to_permuted_idx=indices,
        expert_scale_factor=weights,
        shared_expert_output=shared,
        weight_bias=WEIGHT_BIAS,
    )
    expected = _protocol_prenorm(
        protocol,
        _local_finalize(routed, weights, indices, shared),
        zero_residual,
        group,
        None,
    )
    assert torch.equal(residual_out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize(
    "protocol,large_m",
    (
        ("ll", False),
        ("bt", False),
        ("bt", True),
        ("ht", False),
        # HT's persistent pipeline only reaches steady state over many tokens,
        # and tp=8 HT is new at this hidden size, so cover it past the BT range.
        ("ht", True),
    ),
)
@torch.inference_mode()
def test_protocol_numerical_contract(distributed_group, protocol, large_m):
    group = distributed_group
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    m = _past_bt_split(world_size) if large_m else world_size
    workspace = _workspace(protocol, m, group)
    symmetric_reference = (
        _make_symmetric_reference((m, HIDDEN_SIZE), group) if protocol == "ht" else None
    )

    common_generator = torch.Generator(device="cuda").manual_seed(3200)
    residual = torch.randn(
        m,
        HIDDEN_SIZE,
        generator=common_generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    gamma = torch.randn(
        HIDDEN_SIZE,
        generator=common_generator,
        dtype=torch.bfloat16,
        device="cuda",
    )

    try:
        local = _order_sensitive_local(m, rank, world_size)
        residual_out = torch.empty_like(local)
        norm_out = torch.empty_like(local)
        allreduce_fusion(
            input=local,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=True,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=gamma,
            rms_eps=RMS_EPS,
            weight_bias=WEIGHT_BIAS,
        )
        reference_prenorm = _protocol_prenorm(
            protocol, local, residual, group, symmetric_reference
        )
        _assert_prenorm_contract(protocol, residual_out, reference_prenorm)
        _assert_norm_contract(
            protocol, norm_out, residual_out, reference_prenorm, gamma
        )

        routed = torch.zeros(
            m * TOP_K,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        weights = torch.zeros(m, TOP_K, dtype=torch.bfloat16, device="cuda")
        shared = torch.zeros(m, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        rank_generator = torch.Generator(device="cuda").manual_seed(3300 + rank)
        routed[rank * TOP_K : (rank + 1) * TOP_K].normal_(generator=rank_generator)
        weights[rank].normal_(generator=rank_generator)
        shared[rank].normal_(generator=rank_generator)
        indices = torch.arange(m * TOP_K, dtype=torch.int32, device="cuda").reshape(
            m, TOP_K
        )
        zero_residual = torch.zeros_like(residual)
        residual_out.zero_()
        norm_out.zero_()
        allreduce_fusion(
            input=routed,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
            launch_with_pdl=True,
            residual_in=zero_residual,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=gamma,
            rms_eps=RMS_EPS,
            expanded_idx_to_permuted_idx=indices,
            expert_scale_factor=weights,
            shared_expert_output=shared,
            weight_bias=WEIGHT_BIAS,
        )
        local_finalize = _local_finalize(routed, weights, indices, shared)
        reference_prenorm = _protocol_prenorm(
            protocol,
            local_finalize,
            zero_residual,
            group,
            symmetric_reference,
        )
        assert torch.equal(
            residual_out.view(torch.int16), reference_prenorm.view(torch.int16)
        )
        _assert_norm_contract(
            protocol, norm_out, residual_out, reference_prenorm, gamma
        )
    finally:
        workspace.destroy()


@pytest.mark.parametrize(
    "protocol,large_m", (("ll", False), ("bt", False), ("bt", True))
)
@torch.inference_mode()
def test_disabled_norm_protocol_contract(distributed_group, protocol, large_m):
    group = distributed_group
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    m = _past_bt_split(world_size) if large_m else world_size
    workspace = _workspace(protocol, m, group, apply_rms_norm=False)
    try:
        _check_disabled_norm_pair(protocol, workspace, m, rank, world_size, group)
    finally:
        workspace.destroy()


@torch.inference_mode()
def test_no_norm_config_end_to_end(distributed_group):
    """NO_NORM_CONFIG's own routes, at an M on each side of every bound.

    The per-protocol tests pin one protocol; this goes through the shipped
    norm-free routes, so a bad bound or preset would surface here. The contract
    follows whichever protocol the route picks, and none may be HT.
    """
    group = distributed_group
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    profile = NO_NORM_CONFIG.resolve(
        tp_size=world_size,
        hidden_size=HIDDEN_SIZE,
        top_k=TOP_K,
        dtype=torch.bfloat16,
        capacity_m=8192,
    )
    bounds = sorted(
        {
            bound
            for route in (profile.finalize_routes, profile.all_reduce_routes)
            for bound in route.upper_bounds
            if bound is not None
        }
    )
    m_values = sorted({world_size, *bounds, *(b + 1 for b in bounds)})
    workspace = _workspace(
        None, max(m_values), group, apply_rms_norm=False, config=NO_NORM_CONFIG
    )
    try:
        for m in m_values:
            ar_protocol = profile.all_reduce_routes.select(m).protocol
            finalize_protocol = profile.finalize_routes.select(m).protocol
            assert ProtocolKind.HT not in (ar_protocol, finalize_protocol)
            # The finalize prenorm is bit-exact under either protocol, so only
            # the AR route decides how strict the check is.
            contract = "ll" if ar_protocol is ProtocolKind.LL else "bt"
            _check_disabled_norm_pair(contract, workspace, m, rank, world_size, group)
    finally:
        workspace.destroy()


@torch.inference_mode()
def test_high_throughput_rejects_disabled_norm(distributed_group):
    """HT cannot compile the norm out. Unlike hidden 5120, HT exists at tp=8
    here, so this runs at both rank counts rather than skipping."""
    group = distributed_group
    with pytest.raises(NotImplementedError, match="apply_rms_norm"):
        _workspace("ht", dist.get_world_size(group), group, apply_rms_norm=False)
