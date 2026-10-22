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

"""The MNNVL CuTe DSL backend through the unified all-reduce API.

Workspaces come from ``create_allreduce_fusion_workspace(backend=
"mnnvl-cutedsl")`` with the built-in routing configs, so the token counts below
cross every protocol range of the TP4 hidden-8192 profiles. Run with four
ranks, for example:
    torchrun --standalone --nproc-per-node=4 -m pytest -q \
        tests/comm/test_mnnvl_cutedsl_allreduce_api.py
"""

import os

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import AllReduceFusionPattern as P
from flashinfer.comm import (
    allreduce_fusion,
    create_allreduce_fusion_workspace,
    is_mnnvl_cutedsl_supported,
)

pytestmark = pytest.mark.gpu_4

HIDDEN = 8192
TOP_K = 10
CAPACITY = 2048
RMS_EPS = 1e-5
FP8_MAX = 448.0
MIN_SCALE = (1.0 / 512.0) / FP8_MAX
# One count inside every LL, BT and two-shot range of the GB300 and Rubin
# tables, plus a count above the largest finite bound.
TOKEN_COUNTS = (1, 12, 24, 40, 56, 100, 200, 300, 1025)


def _supported_everywhere(group, **kwargs) -> bool:
    flag = torch.tensor(
        int(
            is_mnnvl_cutedsl_supported(
                dist.get_world_size(group),
                HIDDEN,
                torch.bfloat16,
                max_token_num=CAPACITY,
                top_k=TOP_K,
                **kwargs,
            )
        ),
        device="cuda",
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


@pytest.fixture(scope="module")
def distributed_group():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Run this test with four distributed ranks")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", device_id=device)
    try:
        group = dist.group.WORLD
        if not _supported_everywhere(group, apply_rms_norm=True) or not (
            _supported_everywhere(group, apply_rms_norm=False)
        ):
            pytest.skip("No MNNVL CuTe DSL TP4 hidden-8192 profile on this system")
        yield group
    finally:
        if owns_group:
            dist.destroy_process_group()


def _workspace(group, **options):
    return create_allreduce_fusion_workspace(
        backend="mnnvl-cutedsl",
        world_size=dist.get_world_size(group),
        rank=dist.get_rank(group),
        max_token_num=CAPACITY,
        hidden_dim=HIDDEN,
        dtype=torch.bfloat16,
        group=group,
        top_k=TOP_K,
        rms_eps=RMS_EPS,
        **options,
    )


def _local(m: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed + dist.get_rank())
    return torch.randn(
        m, HIDDEN, generator=generator, dtype=torch.bfloat16, device="cuda"
    )


def _fp32_sum(local: torch.Tensor, group) -> torch.Tensor:
    peers = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, local, group=group)
    return torch.stack(peers).float().sum(0)


def _assert_bf16_of(
    actual: torch.Tensor,
    expected: torch.Tensor,
    ulps: int = 1,
    magnitude: torch.Tensor | None = None,
):
    """``actual`` (BF16) is within ``ulps`` BF16 steps of the FP32 ``expected``.

    Steps are taken at ``magnitude`` (default ``|expected|``), which covers an
    intermediate BF16 rounding of a larger addend.
    """
    assert actual.dtype == torch.bfloat16
    if magnitude is None:
        magnitude = expected.abs()
    step = torch.ldexp(
        torch.ones_like(expected), torch.frexp(magnitude).exponent - 8
    ).clamp(min=2.0**-126)
    error = (actual.float() - expected).abs()
    assert (error <= ulps * step).all(), float((error / step).max())


@torch.inference_mode()
def test_plain_all_reduce(distributed_group):
    group = distributed_group
    workspace = _workspace(group, apply_rms_norm=False, add_residual=False)
    try:
        for m in TOKEN_COUNTS:
            local = _local(m, 100 * m)
            expected = _fp32_sum(local, group)
            returned = allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=P.kAllReduce,
                launch_with_pdl=True,
            )
            _assert_bf16_of(returned, expected)
            output = torch.empty_like(local)
            returned = allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=P.kAllReduce,
                launch_with_pdl=True,
                output=output,
            )
            assert returned.data_ptr() == output.data_ptr()
            _assert_bf16_of(output, expected)
    finally:
        workspace.destroy()


@torch.inference_mode()
def test_plain_all_reduce_rejects_norm_operands(distributed_group):
    group = distributed_group
    workspace = _workspace(group, apply_rms_norm=False, add_residual=False)
    try:
        local = _local(4, 7)
        with pytest.raises(ValueError, match="kAllReduce does not take: residual_in"):
            allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=P.kAllReduce,
                launch_with_pdl=True,
                residual_in=local,
            )
        with pytest.raises(ValueError, match="output"):
            allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=P.kAllReduce,
                launch_with_pdl=True,
                output=torch.empty(5, HIDDEN, dtype=torch.bfloat16, device="cuda"),
            )
    finally:
        workspace.destroy()


@torch.inference_mode()
def test_residual_rms_norm(distributed_group):
    group = distributed_group
    workspace = _workspace(group)
    gamma = torch.randn(
        HIDDEN,
        generator=torch.Generator(device="cuda").manual_seed(11),
        dtype=torch.bfloat16,
        device="cuda",
    )
    try:
        with pytest.raises(ValueError, match="apply_rms_norm=False"):
            allreduce_fusion(
                input=_local(4, 3),
                workspace=workspace,
                pattern=P.kAllReduce,
                launch_with_pdl=True,
            )
        for m in TOKEN_COUNTS:
            local = _local(m, 200 * m)
            residual = torch.randn(
                m,
                HIDDEN,
                generator=torch.Generator(device="cuda").manual_seed(m),
                dtype=torch.bfloat16,
                device="cuda",
            )
            reduced = _fp32_sum(local, group)
            prenorm = reduced + residual.float()
            magnitude = torch.maximum(reduced.abs(), residual.float().abs())
            residual_out = torch.empty_like(local)
            norm_out = torch.empty_like(local)
            returned = allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=P.kARResidualRMSNorm,
                launch_with_pdl=True,
                residual_in=residual,
                rms_gamma=gamma,
                rms_eps=RMS_EPS,
                residual_out=residual_out,
                norm_out=norm_out,
            )
            assert returned.data_ptr() == norm_out.data_ptr()
            # The reduced sum and the residual are each rounded to BF16 once.
            _assert_bf16_of(residual_out, prenorm, ulps=2, magnitude=magnitude)
            values = residual_out.float()
            expected_norm = (
                values
                * torch.rsqrt(values.square().mean(-1, keepdim=True) + RMS_EPS)
                * gamma.float()
            )
            _assert_bf16_of(norm_out, expected_norm, ulps=2)
    finally:
        workspace.destroy()


@torch.inference_mode()
def test_residual_rms_norm_dynamic_fp8(distributed_group):
    group = distributed_group
    workspace = _workspace(group, output_dtype=torch.float8_e4m3fn, dynamic_fp8=True)
    gamma = torch.randn(
        HIDDEN,
        generator=torch.Generator(device="cuda").manual_seed(12),
        dtype=torch.bfloat16,
        device="cuda",
    )
    try:
        for m in TOKEN_COUNTS:
            local = _local(m, 300 * m)
            residual = torch.randn(
                m,
                HIDDEN,
                generator=torch.Generator(device="cuda").manual_seed(m + 1),
                dtype=torch.bfloat16,
                device="cuda",
            )
            reduced = _fp32_sum(local, group)
            prenorm = reduced + residual.float()
            magnitude = torch.maximum(reduced.abs(), residual.float().abs())
            residual_out = torch.empty_like(local)
            quant_out = torch.empty(m, HIDDEN, dtype=torch.float8_e4m3fn, device="cuda")
            scale_out = torch.empty(m, 1, dtype=torch.float32, device="cuda")
            returned = allreduce_fusion(
                input=local,
                workspace=workspace,
                pattern=P.kARResidualRMSNormDynamicFP8Quant,
                launch_with_pdl=True,
                residual_in=residual,
                rms_gamma=gamma,
                rms_eps=RMS_EPS,
                residual_out=residual_out,
                quant_out=quant_out,
                scale_out=scale_out,
            )
            assert returned.data_ptr() == quant_out.data_ptr()
            _assert_bf16_of(residual_out, prenorm, ulps=2, magnitude=magnitude)
            values = residual_out.float()
            norm = (
                values
                * torch.rsqrt(values.square().mean(-1, keepdim=True) + RMS_EPS)
                * gamma.float()
            )
            scale = (norm.abs().amax(-1, keepdim=True) / FP8_MAX).clamp(min=MIN_SCALE)
            torch.testing.assert_close(scale_out, scale, rtol=1e-3, atol=0)
            # Round-to-nearest E4M3 is within half a step: 2**-4 relative for
            # normals, 2**-10 * scale for subnormals, plus approximate rsqrt.
            decoded = quant_out.float() * scale_out
            bound = (2.0**-4 + 1e-3) * norm.abs() + scale_out * (2.0**-10 + 1e-6)
            assert ((decoded - norm).abs() <= bound).all()
    finally:
        workspace.destroy()
