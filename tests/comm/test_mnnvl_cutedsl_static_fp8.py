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

"""Static FP8 output of the MNNVL CuTe DSL all-reduce fusion.

Runs at the launched world size (4, 8 or 16 ranks), for example:
    torchrun --standalone --nproc-per-node=4 -m pytest -q \
        tests/comm/test_mnnvl_cutedsl_static_fp8.py
"""

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm import AllReduceFusionPattern as P
from flashinfer.comm import allreduce_fusion
from flashinfer.comm.mnnvl_cutedsl import (
    KernelTarget,
    MNNVLCuteDSLConfig,
    MRangeDispatch,
    ProtocolKind,
    StaticProfile,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_bt import (
    BTAllReduceTuning,
    BTFinalizeTuning,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_ht import (
    HTAllReduceTuning,
    HTFinalizeTuning,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_ll import (
    LLAllReduceTuning,
    LLFinalizeTuning,
)
from flashinfer.comm.mnnvl_cutedsl_ar import MNNVLCuteDSLAllReduceFusionWorkspace
from tests.comm.test_mnnvl_cutedsl_numerical_contract import (
    RMS_EPS,
    WEIGHT_BIAS,
    _assert_norm_contract,
    _assert_prenorm_contract,
    _local_finalize,
    _make_symmetric_reference,
    _protocol_prenorm,
    distributed_group as distributed_group,
)

pytestmark = pytest.mark.gpu_4

CAPACITY = 64
FP8_MAX = 448.0


def _tunings(protocol: str, hidden_size: int):
    """Finalize and all-reduce tunings valid at TP4, TP8 and TP16.

    The protocol defaults cover both tested shapes, except that HT's default
    512-thread x 2-vector shard only tiles multiples of 8192 elements; H3584
    uses the 448 x 1 shard of the shipped Kimi K3 HT presets.
    """
    if protocol == "ll":
        return ProtocolKind.LL, LLFinalizeTuning(), LLAllReduceTuning()
    if protocol == "bt":
        return ProtocolKind.BT, BTFinalizeTuning(), BTAllReduceTuning()
    if hidden_size % 8192 == 0:
        return ProtocolKind.HT, HTFinalizeTuning(), HTAllReduceTuning()
    shard = dict(consumer_threads=448, vectors_per_thread=1)
    return ProtocolKind.HT, HTFinalizeTuning(**shard), HTAllReduceTuning(**shard)


def _single_protocol_config(
    protocol: str, tp_size: int, hidden_size: int, top_k: int
) -> MNNVLCuteDSLConfig:
    kind, finalize, all_reduce = _tunings(protocol, hidden_size)

    def route(preset):
        return MRangeDispatch(
            upper_bounds=(None,), targets=(KernelTarget(kind, preset),)
        )

    return MNNVLCuteDSLConfig(
        profiles=(
            StaticProfile(
                tp_size=tp_size,
                hidden_size=hidden_size,
                top_k=top_k,
                dtype=torch.bfloat16,
                finalize_routes=route(finalize),
                all_reduce_routes=route(all_reduce),
            ),
        )
    )


def _expected_fp8(norm: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # The kernel multiplies by an IEEE reciprocal and saturates, as TRT-LLM does.
    scaled = norm.float() * scale.float().reshape(1).reciprocal()
    return scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


# Each protocol covers both write_norm_output modes and both shapes (Qwen3.8
# H8192/top-k 10 and Kimi K3 H3584/top-k 16) without a full cross product.
@pytest.mark.parametrize(
    "protocol,write_norm_output,hidden_size,top_k",
    [
        ("ll", False, 8192, 10),
        ("ll", True, 3584, 16),
        ("bt", False, 3584, 16),
        ("bt", True, 8192, 10),
        ("ht", False, 8192, 10),
        ("ht", True, 3584, 16),
    ],
)
@torch.inference_mode()
def test_static_fp8_outputs_and_replay(
    distributed_group,  # noqa: F811 - pytest injects the imported fixture.
    protocol,
    write_norm_output,
    hidden_size,
    top_k,
):
    group = distributed_group
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    common = dict(
        tp_size=world_size,
        tp_rank=rank,
        max_token_num=CAPACITY,
        hidden_dim=hidden_size,
        dtype=torch.bfloat16,
        group=group,
        top_k=top_k,
        rms_eps=RMS_EPS,
        weight_bias=WEIGHT_BIAS,
        config=_single_protocol_config(protocol, world_size, hidden_size, top_k),
    )
    baseline = MNNVLCuteDSLAllReduceFusionWorkspace(**common)
    workspace = MNNVLCuteDSLAllReduceFusionWorkspace(
        **common,
        output_dtype=torch.float8_e4m3fn,
        write_norm_output=write_norm_output,
    )
    # Patterns 4 and 12 need opposite write_norm_output settings, so the
    # finalize half runs on the quant-only workspace and is rejected otherwise.
    finalize_cases = (False,) if write_norm_output else (False, True)
    try:
        for finalize in finalize_cases:
            for m in (1, 9, 61):
                # Residual/gamma are replicated; expert contributions vary by rank.
                replicated = torch.Generator(device="cuda").manual_seed(7100 + m)
                local_rng = torch.Generator(device="cuda").manual_seed(7200 + rank)
                residual = torch.randn(
                    m,
                    hidden_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=replicated,
                )
                gamma = torch.randn(
                    hidden_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=replicated,
                )
                rows = m * top_k + 17 if finalize else m
                source = torch.randn(
                    rows,
                    hidden_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=local_rng,
                )
                moe = {}
                if finalize:
                    moe = dict(
                        expanded_idx_to_permuted_idx=torch.randint(
                            rows,
                            (m, top_k),
                            device="cuda",
                            dtype=torch.int32,
                            generator=local_rng,
                        ),
                        expert_scale_factor=torch.rand(
                            m,
                            top_k,
                            device="cuda",
                            dtype=torch.bfloat16,
                            generator=local_rng,
                        ),
                        shared_expert_output=torch.randn(
                            m,
                            hidden_size,
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
                    (m, hidden_size), device="cuda", dtype=torch.float8_e4m3fn
                )
                # One configuration passes a 0-d scale; the kernel sees a
                # one-element view of the same storage either way.
                scale = torch.tensor(
                    0.37 if write_norm_output else [0.37],
                    device="cuda",
                    dtype=torch.float32,
                )
                baseline_pattern = (
                    P.kMoEFinalizeARResidualRMSNorm
                    if finalize
                    else P.kARResidualRMSNorm
                )
                quant_pattern = (
                    P.kMoEFinalizeARResidualRMSNormFP8Quant
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
                assert torch.equal(
                    quant.view(torch.uint8),
                    _expected_fp8(reference_norm, scale).view(torch.uint8),
                )
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
                    _make_symmetric_reference((m, hidden_size), group)
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
                # Inexact reciprocals (0.37, 3.0) catch an approximate divide;
                # 1/256 drives most of the norm past the E4M3 range.
                for value in (3.0, 1 / 256, 0.37, 3.0, 1 / 256, 0.37):
                    # Change payload and scale, and wrap the three mailbox generations.
                    source.neg_()
                    scale.fill_(value)
                    reference()
                    quant.view(torch.uint8).fill_(0x7F)
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = _expected_fp8(reference_norm, scale)
                    assert torch.equal(
                        quant.view(torch.uint8), expected.view(torch.uint8)
                    )
                    if value == 1 / 256:
                        decoded = quant.float()
                        assert (decoded == FP8_MAX).any()
                        assert (decoded == -FP8_MAX).any()
                    assert torch.equal(
                        prenorm.view(torch.int16), reference_residual.view(torch.int16)
                    )
                    if norm is not None:
                        assert torch.equal(
                            norm.view(torch.int16), reference_norm.view(torch.int16)
                        )

                # A Python float and a non-FP32 tensor are converted, like the
                # TRT-LLM and MNNVL backends do; the result must not change.
                scale.fill_(0.37)
                reference()
                expected = _expected_fp8(reference_norm, scale)
                for converted in (
                    0.37,
                    torch.tensor([0.37], device="cuda", dtype=torch.float64),
                ):
                    quant.view(torch.uint8).fill_(0x7F)
                    allreduce_fusion(**dict(quant_args, scale_factor=converted))
                    assert torch.equal(
                        quant.view(torch.uint8), expected.view(torch.uint8)
                    )

                for invalid in (
                    None,
                    torch.ones(2, device="cuda"),
                    torch.ones(1),
                ):
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
