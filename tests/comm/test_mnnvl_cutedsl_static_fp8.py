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

"""Unrun GPU validation for the source-only static FP8 reference.

Run on a supported eight/sixteen-rank NVLink system after preparing that system:
    torchrun --standalone --nproc-per-node=8 -m pytest -q \
        tests/comm/test_mnnvl_cutedsl_static_fp8.py

This file has only been syntax-checked; it is not evidence of GPU correctness.
"""

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import AllReduceFusionPattern as P
from flashinfer.comm import allreduce_fusion
from flashinfer.comm.mnnvl_cutedsl_ar import MNNVLCuteDSLAllReduceFusionWorkspace
from tests.comm.test_mnnvl_cutedsl_numerical_contract import (
    HIDDEN_SIZE,
    PROTOCOL_CONFIGS,
    RMS_EPS,
    TOP_K,
    WEIGHT_BIAS,
    _assert_norm_contract,
    _assert_prenorm_contract,
    _local_finalize,
    _make_symmetric_reference,
    _protocol_prenorm,
    distributed_group as distributed_group,
)

pytestmark = [pytest.mark.gpu_8, pytest.mark.arch_blackwell]


@pytest.mark.parametrize("protocol", ("ll", "bt", "ht"))
@pytest.mark.parametrize("write_norm_output", (False, True))
@torch.inference_mode()
def test_static_fp8_outputs_and_replay(
    distributed_group,  # noqa: F811 - pytest injects the imported fixture.
    protocol,
    write_norm_output,
):
    group = distributed_group
    rank = dist.get_rank(group)
    common = dict(
        tp_size=dist.get_world_size(group),
        tp_rank=rank,
        max_token_num=33,
        hidden_dim=HIDDEN_SIZE,
        dtype=torch.bfloat16,
        group=group,
        top_k=TOP_K,
        rms_eps=RMS_EPS,
        weight_bias=WEIGHT_BIAS,
        config=PROTOCOL_CONFIGS[protocol],
    )
    baseline = MNNVLCuteDSLAllReduceFusionWorkspace(**common)
    workspace = MNNVLCuteDSLAllReduceFusionWorkspace(
        **common,
        output_dtype=torch.float8_e4m3fn,
        write_norm_output=write_norm_output,
    )
    try:
        for finalize in (False, True):
            for m in (1, 7, 33):
                # Residual/gamma are replicated; expert contributions vary by rank.
                replicated = torch.Generator(device="cuda").manual_seed(7100 + m)
                local_rng = torch.Generator(device="cuda").manual_seed(7200 + rank)
                residual = torch.randn(
                    m,
                    HIDDEN_SIZE,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=replicated,
                )
                gamma = torch.randn(
                    HIDDEN_SIZE,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=replicated,
                )
                rows = m * TOP_K + 17 if finalize else m
                source = torch.randn(
                    rows,
                    HIDDEN_SIZE,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=local_rng,
                )
                moe = {}
                if finalize:
                    moe = dict(
                        expanded_idx_to_permuted_idx=torch.randint(
                            rows,
                            (m, TOP_K),
                            device="cuda",
                            dtype=torch.int32,
                            generator=local_rng,
                        ),
                        expert_scale_factor=torch.rand(
                            m,
                            TOP_K,
                            device="cuda",
                            dtype=torch.bfloat16,
                            generator=local_rng,
                        ),
                        shared_expert_output=torch.randn(
                            m,
                            HIDDEN_SIZE,
                            device="cuda",
                            dtype=torch.bfloat16,
                            generator=local_rng,
                        ),
                    )
                reference_norm = torch.empty_like(residual)
                reference_residual = torch.empty_like(residual)
                prenorm = torch.empty_like(residual)
                norm = torch.empty_like(residual) if write_norm_output else None
                quant = torch.empty(
                    (m, HIDDEN_SIZE), device="cuda", dtype=torch.float8_e4m3fn
                )
                scale = torch.tensor([0.5], device="cuda", dtype=torch.float32)
                baseline_pattern = (
                    P.kMoEFinalizeARResidualRMSNorm
                    if finalize
                    else P.kARResidualRMSNorm
                )
                quant_pattern = (
                    P.kMoEFinalizeARResidualRMSNorm
                    if finalize
                    else P.kARResidualRMSNormOutFP8Quant
                    if write_norm_output
                    else P.kARResidualRMSNormFP8Quant
                )
                args = dict(
                    input=source,
                    residual_in=residual,
                    rms_gamma=gamma,
                    rms_eps=RMS_EPS,
                    weight_bias=WEIGHT_BIAS,
                    launch_with_pdl=True,
                    **moe,
                )
                quant_args = dict(
                    **args,
                    workspace=workspace,
                    pattern=quant_pattern,
                    residual_out=prenorm,
                    norm_out=norm,
                    quant_out=quant,
                    scale_factor=scale,
                )

                def reference():
                    allreduce_fusion(
                        **args,
                        workspace=baseline,
                        pattern=baseline_pattern,
                        residual_out=reference_residual,
                        norm_out=reference_norm,
                    )

                reference()
                assert allreduce_fusion(**quant_args).data_ptr() == quant.data_ptr()
                local = (
                    _local_finalize(
                        source,
                        moe["expert_scale_factor"],
                        moe["expanded_idx_to_permuted_idx"],
                        moe["shared_expert_output"],
                    )
                    if finalize
                    else source
                )
                symmetric = (
                    _make_symmetric_reference((m, HIDDEN_SIZE), group)
                    if protocol == "ht"
                    else None
                )
                independent_prenorm = _protocol_prenorm(
                    protocol, local, residual, group, symmetric
                )
                _assert_prenorm_contract(protocol, prenorm, independent_prenorm)
                _assert_norm_contract(
                    protocol,
                    reference_norm,
                    reference_residual,
                    independent_prenorm,
                    gamma,
                )

                # Warm up on a side stream before capture. All buffers are supplied.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        allreduce_fusion(**quant_args)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                dist.barrier(group=group)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    allreduce_fusion(**quant_args)
                for value in (0.5, 1 / 256, 2.0, 0.5, 1 / 256, 2.0):
                    # Change payload and scale, and wrap the three mailbox generations.
                    source.neg_()
                    scale.fill_(value)
                    reference()
                    quant.view(torch.uint8).fill_(0x7F)
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = (reference_norm.float() * scale.reciprocal()).clamp(
                        -448, 448
                    )
                    expected = expected.to(torch.float8_e4m3fn)
                    assert torch.equal(
                        quant.view(torch.uint8), expected.view(torch.uint8)
                    )
                    assert torch.equal(
                        prenorm.view(torch.int16), reference_residual.view(torch.int16)
                    )
                    if norm is not None:
                        assert torch.equal(
                            norm.view(torch.int16), reference_norm.view(torch.int16)
                        )

                for invalid in (None, 0.5, torch.ones(2, device="cuda")):
                    with pytest.raises(ValueError, match="scale_factor"):
                        allreduce_fusion(**dict(quant_args, scale_factor=invalid))
                with pytest.raises(ValueError, match="quant_out must have dtype"):
                    allreduce_fusion(
                        **dict(quant_args, quant_out=torch.empty_like(residual))
                    )
                with pytest.raises(NotImplementedError, match="Unsupported"):
                    allreduce_fusion(
                        **dict(quant_args, pattern=P.kARResidualRMSNormDynamicFP8Quant)
                    )
                torch.cuda.synchronize()
                dist.barrier(group=group)
                del graph, symmetric
    finally:
        torch.cuda.synchronize()
        dist.barrier(group=group)
        workspace.destroy()
        baseline.destroy()
