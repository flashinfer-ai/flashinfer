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

"""Dynamic per-token FP8 output of the MNNVL CuTe DSL all-reduce fusion.

Runs at the launched world size (4, 8 or 16 ranks), for example:
    torchrun --standalone --nproc-per-node=4 -m pytest -q \
        tests/comm/test_mnnvl_cutedsl_dynamic_fp8.py
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
    BT_ALL_REDUCE_RUBIN_TP4_H8192_FUSED,
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
    distributed_group as distributed_group,
)

pytestmark = pytest.mark.gpu_4

CAPACITY = 64
TOP_K = 10
FP8_MAX = 448.0
MIN_SCALE = (1.0 / 512.0) / FP8_MAX


def _tunings(protocol: str, hidden_size: int):
    """Finalize and all-reduce tunings valid at TP4, TP8 and TP16.

    ``ht_pipelined`` exercises the HT RMS path that stages three tokens per
    group in shared memory.
    """
    if protocol == "ll":
        return ProtocolKind.LL, LLFinalizeTuning(), LLAllReduceTuning()
    if protocol == "bt":
        return ProtocolKind.BT, BTFinalizeTuning(), BTAllReduceTuning()
    if protocol == "bt_fused":
        return (
            ProtocolKind.BT,
            BTFinalizeTuning(),
            BT_ALL_REDUCE_RUBIN_TP4_H8192_FUSED,
        )
    shard = (
        {}
        if hidden_size % 8192 == 0
        else dict(consumer_threads=448, vectors_per_thread=1)
    )
    all_reduce = dict(shard)
    if protocol == "ht_pipelined":
        all_reduce.update(
            stages=7 if hidden_size % 8192 == 0 else 10, rms_pipeline_stages=3
        )
    return (
        ProtocolKind.HT,
        HTFinalizeTuning(**shard),
        HTAllReduceTuning(**all_reduce),
    )


def _config(protocol: str, tp_size: int, hidden_size: int) -> MNNVLCuteDSLConfig:
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
                top_k=TOP_K,
                dtype=torch.bfloat16,
                finalize_routes=route(finalize),
                all_reduce_routes=route(all_reduce),
            ),
        )
    )


def _reference(prenorm: torch.Tensor, gamma: torch.Tensor):
    """FP32 norm, per-token scale and E4M3 values of the TRT-LLM recipe."""
    values = prenorm.float()
    norm = (
        values
        * torch.rsqrt(values.square().mean(-1, keepdim=True) + RMS_EPS)
        * (gamma.float() + WEIGHT_BIAS)
    )
    scale = (norm.abs().amax(-1, keepdim=True) / FP8_MAX).clamp(min=MIN_SCALE)
    quant = (norm / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return norm, scale, quant


def _assert_dynamic_fp8(quant, scale_out, prenorm, gamma):
    norm, scale, expected = _reference(prenorm, gamma)
    torch.testing.assert_close(scale_out, scale, rtol=1e-3, atol=0)
    # Round-to-nearest E4M3 is within half a step: 2**-4 relative for normals,
    # 2**-10 * scale in the subnormal range. The slack covers the kernel's
    # approximate rsqrt and summation order.
    decoded = quant.float() * scale_out
    bound = (2.0**-4 + 1e-3) * norm.abs() + scale_out * (2.0**-10 + 1e-6)
    assert ((decoded - norm).abs() <= bound).all()
    # Every nonzero row reaches the saturation value at its amax element.
    nonzero = norm.abs().amax(-1) > 0
    assert (quant.float().abs().amax(-1)[nonzero] == FP8_MAX).all()
    # Ties may round the other way under a slightly different scale, but
    # nearly every byte matches the reference recipe exactly.
    mismatched = (quant.view(torch.uint8) != expected.view(torch.uint8)).float()
    assert mismatched.mean().item() < 1e-3


@pytest.mark.parametrize(
    "protocol,write_norm_output,hidden_size",
    [
        ("ll", False, 8192),
        ("ll", True, 3584),
        ("bt", False, 8192),
        ("bt", True, 3584),
        ("bt_fused", False, 8192),
        ("bt_fused", True, 3584),
        ("ht", False, 8192),
        ("ht", True, 3584),
        ("ht_pipelined", False, 8192),
        ("ht_pipelined", True, 3584),
    ],
)
@torch.inference_mode()
def test_dynamic_fp8_outputs_and_replay(
    distributed_group,  # noqa: F811 - pytest injects the imported fixture.
    protocol,
    write_norm_output,
    hidden_size,
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
        top_k=TOP_K,
        rms_eps=RMS_EPS,
        weight_bias=WEIGHT_BIAS,
        config=_config(protocol, world_size, hidden_size),
    )
    baseline = MNNVLCuteDSLAllReduceFusionWorkspace(**common)
    workspace = MNNVLCuteDSLAllReduceFusionWorkspace(
        **common,
        output_dtype=torch.float8_e4m3fn,
        write_norm_output=write_norm_output,
        dynamic_fp8=True,
    )
    pattern = (
        P.kARResidualRMSNormOutDynamicFP8Quant
        if write_norm_output
        else P.kARResidualRMSNormDynamicFP8Quant
    )
    try:
        for m in (1, 9, 61):
            replicated = torch.Generator(device="cuda").manual_seed(8100 + m)
            local_rng = torch.Generator(device="cuda").manual_seed(8200 + rank)
            residual = torch.randn(
                m,
                hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
                generator=replicated,
            )
            gamma = torch.randn(
                hidden_size, device="cuda", dtype=torch.bfloat16, generator=replicated
            )
            source = torch.randn(
                m, hidden_size, device="cuda", dtype=torch.bfloat16, generator=local_rng
            )
            # Row 0 sums to an exact zero: the scale falls back to its minimum.
            residual[0].zero_()
            source[0].zero_()
            if m > 2:
                # One outlier per row on a different rank and column, so the
                # amax lives in a different CTA, cluster peer or RMS warp.
                rows = torch.arange(1, m, device="cuda")
                columns = (rows * 977 + rank * 131) % hidden_size
                source[rows, columns] = 64.0 * (rank + 1)
            reference_residual = torch.empty_like(residual)
            reference_norm = torch.empty_like(residual)
            prenorm = torch.empty_like(residual)
            norm = torch.empty_like(residual) if write_norm_output else None
            quant = torch.empty(
                (m, hidden_size), device="cuda", dtype=torch.float8_e4m3fn
            )
            scale_out = torch.empty((m, 1), device="cuda", dtype=torch.float32)
            args = dict(
                input=source,
                residual_in=residual,
                rms_gamma=gamma,
                rms_eps=RMS_EPS,
                weight_bias=WEIGHT_BIAS,
                launch_with_pdl=True,
            )
            quant_args = dict(
                **args,
                workspace=workspace,
                pattern=pattern,
                residual_out=prenorm,
                norm_out=norm,
                quant_out=quant,
                scale_out=scale_out,
            )

            def reference():
                allreduce_fusion(
                    **args,
                    workspace=baseline,
                    pattern=P.kARResidualRMSNorm,
                    residual_out=reference_residual,
                    norm_out=reference_norm,
                )

            def check():
                # Same reduction kernels: the pre-norm and BF16 norm are exact.
                assert torch.equal(
                    prenorm.view(torch.int16), reference_residual.view(torch.int16)
                )
                if norm is not None:
                    assert torch.equal(
                        norm.view(torch.int16), reference_norm.view(torch.int16)
                    )
                _assert_dynamic_fp8(quant, scale_out, prenorm, gamma)
                assert scale_out[0, 0].item() == pytest.approx(MIN_SCALE, rel=1e-6)
                # A zero row quantizes to signed zeros (the sign of gamma).
                assert (quant[0].float() == 0).all()

            reference()
            assert allreduce_fusion(**quant_args).data_ptr() == quant.data_ptr()
            check()

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
            for factor in (-1.0, 0.125, -8.0, 1.0):
                # Change payload magnitude, and wrap the mailbox generations.
                source.mul_(factor)
                reference()
                quant.view(torch.uint8).fill_(0x7F)
                scale_out.fill_(-1.0)
                graph.replay()
                torch.cuda.synchronize()
                check()

            with pytest.raises(ValueError, match="scale_out is required"):
                allreduce_fusion(**dict(quant_args, scale_out=None))
            with pytest.raises(ValueError, match="scale_out must have shape"):
                allreduce_fusion(**dict(quant_args, scale_out=scale_out.view(-1)))
            with pytest.raises(ValueError, match="scale_factor must be None"):
                allreduce_fusion(**dict(quant_args, scale_factor=1.0))
            with pytest.raises(ValueError, match="dynamic_fp8=False"):
                allreduce_fusion(
                    **dict(
                        quant_args,
                        pattern=P.kARResidualRMSNormFP8Quant,
                        scale_out=None,
                        scale_factor=1.0,
                    )
                )
            torch.cuda.synchronize()
            dist.barrier(group=group)
            del graph
    finally:
        torch.cuda.synchronize()
        dist.barrier(group=group)
        workspace.destroy()
        baseline.destroy()


def test_dynamic_fp8_requires_fp8_output(distributed_group):  # noqa: F811
    group = distributed_group
    with pytest.raises(ValueError, match="dynamic_fp8 requires"):
        MNNVLCuteDSLAllReduceFusionWorkspace(
            tp_size=dist.get_world_size(group),
            tp_rank=dist.get_rank(group),
            max_token_num=CAPACITY,
            hidden_dim=8192,
            dtype=torch.bfloat16,
            group=group,
            dynamic_fp8=True,
        )
