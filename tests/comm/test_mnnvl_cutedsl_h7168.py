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

"""Kimi K3 H7168/K16 numerical and graph-replay coverage."""

import os

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion
from flashinfer.comm.mnnvl_cutedsl import (
    BT_ONLY_CONFIG,
    HT_ONLY_CONFIG,
    LL_ONLY_CONFIG,
)
from flashinfer.comm.mnnvl_cutedsl_ar import MNNVLCuteDSLAllReduceFusionWorkspace
from flashinfer.utils import is_sm100a_supported


HIDDEN_SIZE = 7168
TOP_K = 16
RMS_EPS = 1e-5
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
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Run this test with four distributed ranks")

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


def _reference(
    local: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    reduced = local.float().clone()
    dist.all_reduce(reduced)
    prenorm = (reduced + residual.float()).to(torch.bfloat16)
    values = prenorm.float()
    normalized = values * torch.rsqrt(
        values.square().mean(dim=-1, keepdim=True) + RMS_EPS
    )
    return prenorm, (normalized * gamma.float()).to(torch.bfloat16)


def _assert_outputs(
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    reference: tuple[torch.Tensor, torch.Tensor],
) -> None:
    reference_residual, reference_norm = reference
    torch.testing.assert_close(
        residual_out.float(), reference_residual.float(), rtol=0.04, atol=0.08
    )
    torch.testing.assert_close(
        norm_out.float(), reference_norm.float(), rtol=0.04, atol=0.08
    )


@pytest.mark.parametrize("protocol", ("ll", "bt", "ht"))
@pytest.mark.parametrize("operation", ("all_reduce", "moe_finalize"))
@torch.inference_mode()
def test_h7168_k16_protocols(distributed_group, protocol, operation):
    group = distributed_group
    rank = dist.get_rank(group)
    m = 8
    workspace = MNNVLCuteDSLAllReduceFusionWorkspace(
        tp_size=4,
        tp_rank=rank,
        max_token_num=m,
        hidden_dim=HIDDEN_SIZE,
        dtype=torch.bfloat16,
        group=group,
        top_k=TOP_K,
        rms_eps=RMS_EPS,
        weight_bias=0.0,
        config=PROTOCOL_CONFIGS[protocol],
    )

    common_generator = torch.Generator(device="cuda").manual_seed(7168)
    rank_generator = torch.Generator(device="cuda").manual_seed(7168 + rank + 1)
    residual = torch.randn(
        m,
        HIDDEN_SIZE,
        generator=common_generator,
        dtype=torch.bfloat16,
        device="cuda",
    ).mul_(0.25)
    gamma = torch.randn(
        HIDDEN_SIZE,
        generator=common_generator,
        dtype=torch.bfloat16,
        device="cuda",
    ).mul_(0.25)
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(residual)

    if operation == "all_reduce":
        local = torch.randn(
            m,
            HIDDEN_SIZE,
            generator=rank_generator,
            dtype=torch.bfloat16,
            device="cuda",
        ).mul_(0.25)

        def invoke():
            return allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=AllReduceFusionPattern.kARResidualRMSNorm,
                launch_with_pdl=True,
                residual_in=residual,
                residual_out=residual_out,
                norm_out=norm_out,
                rms_gamma=gamma,
                rms_eps=RMS_EPS,
                weight_bias=0.0,
            )

    else:
        routed = torch.randn(
            m * TOP_K,
            HIDDEN_SIZE,
            generator=rank_generator,
            dtype=torch.bfloat16,
            device="cuda",
        ).mul_(0.25)
        weights = torch.rand(
            m,
            TOP_K,
            generator=rank_generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        weights.div_(weights.float().sum(dim=1, keepdim=True).to(torch.bfloat16))
        indices = torch.arange(m * TOP_K, dtype=torch.int32, device="cuda").reshape(
            m, TOP_K
        )
        shared = torch.randn(
            m,
            HIDDEN_SIZE,
            generator=rank_generator,
            dtype=torch.bfloat16,
            device="cuda",
        ).mul_(0.25)
        local = (
            routed.view(m, TOP_K, HIDDEN_SIZE).float() * weights.float().unsqueeze(-1)
        ).sum(dim=1)
        local.add_(shared.float())
        local = local.to(torch.bfloat16)

        def invoke():
            return allreduce_fusion(
                input=routed,
                workspace=workspace,
                pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
                launch_with_pdl=True,
                residual_in=residual,
                residual_out=residual_out,
                norm_out=norm_out,
                rms_gamma=gamma,
                rms_eps=RMS_EPS,
                expanded_idx_to_permuted_idx=indices,
                expert_scale_factor=weights,
                shared_expert_output=shared,
                weight_bias=0.0,
            )

    try:
        invoke()
        torch.cuda.synchronize()
        reference = _reference(local, residual, gamma)
        _assert_outputs(residual_out, norm_out, reference)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        _assert_outputs(residual_out, norm_out, reference)
    finally:
        workspace.destroy()
