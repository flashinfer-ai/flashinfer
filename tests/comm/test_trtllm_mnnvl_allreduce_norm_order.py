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

"""Numerics of the MNNVL two-shot fused all-reduce + residual + RMSNorm epilogue.

The two-shot kernel normalizes with ``x * rcp(rms) * (weight_bias + gamma)``, the
association of the MNNVL one-shot kernel, the TRT-LLM fusion and the standalone norms.

torchrun --nproc-per-node 4 -m pytest tests/comm/test_trtllm_mnnvl_allreduce_norm_order.py
"""

import os

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import (
    AllReduceFusionPattern,
    allreduce_fusion,
    create_allreduce_fusion_workspace,
)
from flashinfer.comm.comm_backend import TorchDistBackend
from flashinfer.comm.mnnvl import is_multicast_supported


RMS_EPS = 1e-6
pytestmark = pytest.mark.gpu_2


@pytest.fixture(scope="module")
def distributed_group():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        pytest.skip("Run this test with at least two distributed ranks")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not is_multicast_supported(local_rank):
        pytest.skip("NVLink multicast is required")
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", device_id=device)
    try:
        yield dist.group.WORLD
    finally:
        if owns_group:
            dist.destroy_process_group()


def _fused_norm(backend, x, residual, gamma, weight_bias, **kwargs):
    """Run one AllReduce + Residual + RMSNorm on a fresh workspace of ``backend``."""
    workspace = create_allreduce_fusion_workspace(
        backend=backend,
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
        max_token_num=x.shape[0],
        hidden_dim=x.shape[1],
        dtype=x.dtype,
        comm_backend=TorchDistBackend(),
    )
    try:
        norm_out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        allreduce_fusion(
            input=x,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kARResidualRMSNorm,
            norm_out=norm_out,
            residual_out=residual_out,
            residual_in=residual,
            rms_gamma=gamma,
            rms_eps=RMS_EPS,
            weight_bias=weight_bias,
            **kwargs,
        )
        torch.cuda.synchronize()
        dist.barrier()
        return norm_out, residual_out
    finally:
        workspace.destroy()


@pytest.mark.parametrize("weight_bias", [0.0, 1.0])
@pytest.mark.parametrize("hidden_size", [2048, 4096, 7168])
def test_twoshot_norm_multiplies_gamma_last(
    distributed_group, hidden_size, weight_bias
):
    """Gamma multiplies the normalized input last.

    Rank 0 contributes the input; the other ranks and the residual are zero. Column 0
    is 1.0 and every other column is +-21/16 or +-27/16, so each square, and each
    row's sum of squares, is exact in any reduction order and rcp(rms) is one value r.
    Column 0 has ``weight_bias + gamma == 1`` and returns r itself (FP32 output).
    Every other column must equal ``(x * r) * (weight_bias + gamma)``, rounded after
    each product, for FP32 and BF16 I/O. ``x * r`` is inexact here, so the previous
    ``((weight_bias + gamma) * x) * r`` rounds differently on many FP32 elements.
    """
    rows = 64
    generator = torch.Generator(device="cuda").manual_seed(2026)
    sign = torch.where(
        torch.rand(hidden_size, generator=generator, device="cuda") < 0.5, -1.0, 1.0
    )
    magnitude = torch.where(
        torch.rand(hidden_size, generator=generator, device="cuda") < 0.5,
        21.0 / 16.0,
        27.0 / 16.0,
    )
    x_row = sign * magnitude
    x_row[0] = 1.0
    # Non-dyadic gammas, representable in BF16 so both runs read the same values.
    gamma = (torch.rand(hidden_size, generator=generator, device="cuda") + 0.5).to(
        torch.bfloat16
    )
    gamma[0] = 1.0 - weight_bias

    expected = None
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.zeros(rows, hidden_size, device="cuda", dtype=dtype)
        if dist.get_rank() == 0:
            x[:] = x_row
        norm_out, residual_out = _fused_norm(
            "mnnvl",
            x,
            torch.zeros_like(x),
            gamma.to(dtype),
            weight_bias,
            use_oneshot=False,
        )
        assert torch.equal(residual_out.float(), x_row.expand(rows, -1))
        if expected is None:
            # r comes from the FP32 run. It does not depend on the output dtype, because
            # every element of the reduced input is exact in both.
            r = norm_out[:, :1].float()
            mean_square = (x_row.double() ** 2).mean().item()
            torch.testing.assert_close(
                r,
                torch.full_like(r, (mean_square + RMS_EPS) ** -0.5),
                rtol=1e-6,
                atol=0,
            )
            expected = (x_row * r) * (weight_bias + gamma.float())
        torch.testing.assert_close(
            norm_out, expected.to(dtype), rtol=0, atol=0, equal_nan=False
        )


@pytest.mark.parametrize("weight_bias", [0.0, 1.0])
@pytest.mark.parametrize("rows", [36, 64, 128, 252, 256, 512, 1024, 2048])
def test_twoshot_matches_trtllm_fusion(distributed_group, rows, weight_bias):
    """MNNVL two-shot and the TRT-LLM fusion produce identical BF16 outputs.

    With fp32_acc both reduce the ranks in order in FP32, round the residual sum to
    BF16, square in FP32 and normalize with the same association. The TRT-LLM one-shot
    sizes its grid from the row count like the MNNVL two-shot, so from 19 rows up they
    split each row's sum of squares across the same CTAs, and the norm and residual
    outputs must match bit for bit.
    """
    if dist.get_world_size() not in (2, 4):
        pytest.skip("this parity is only established at 2 and 4 ranks")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the shared CTA split is only established on SM100/SM103/SM107")
    hidden_size = 4096
    rank = dist.get_rank()
    shared = torch.Generator(device="cuda").manual_seed(7)
    own = torch.Generator(device="cuda").manual_seed(1234 + rank)
    gamma = (torch.rand(hidden_size, generator=shared, device="cuda") + 0.5).to(
        torch.bfloat16
    )
    x = torch.randn(rows, hidden_size, generator=own, device="cuda").to(torch.bfloat16)
    residual = torch.randn(rows, hidden_size, generator=shared, device="cuda").to(
        torch.bfloat16
    )

    mnnvl_norm, mnnvl_residual = _fused_norm(
        "mnnvl", x, residual, gamma, weight_bias, use_oneshot=False
    )
    trtllm_norm, trtllm_residual = _fused_norm(
        "trtllm", x, residual, gamma, weight_bias, fp32_acc=True, use_oneshot=True
    )
    assert torch.equal(mnnvl_residual, trtllm_residual)
    assert torch.equal(mnnvl_norm, trtllm_norm)
