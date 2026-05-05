# Copyright (c) 2025 by FlashInfer team.
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

"""TraceTemplates for distributed communication ops."""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


@torch.no_grad()
def _allreduce_fusion_reference(
    input: torch.Tensor,
    workspace,
    pattern: int,
    launch_with_pdl: bool = False,
    trigger_completion_at_end: bool = True,
    output=None,
    residual_out=None,
    norm_out=None,
    quant_out=None,
    scale_out=None,
    residual_in=None,
    rms_gamma=None,
    rms_eps: float = 1e-6,
    **_unused,
):
    """Single-rank reference for allreduce_fusion.

    AllReduce is a no-op in a single-process reference (the kernel under
    test normally sums across ranks). This reference therefore treats
    ``input`` as the already-reduced tensor and models the *fusion* side
    of the op:

    - pattern 0 (kAllReduce): passthrough input.
    - pattern 1 (kARResidualRMSNorm): ``residual_out = input + residual_in``;
      ``norm_out = rmsnorm(residual_out, rms_gamma, rms_eps)``.

    Quantized / MoE patterns (>= 2) are outside the single-rank scope —
    this reference raises ``NotImplementedError`` for them and callers
    should exercise the real multi-rank kernel for coverage.
    """
    if pattern == 0:
        out = input.clone()
        if output is not None:
            output.copy_(out)
        return out
    if pattern == 1:
        if residual_in is None or rms_gamma is None:
            raise ValueError(
                "pattern=1 (kARResidualRMSNorm) requires residual_in and rms_gamma"
            )
        pre = input.to(torch.float32) + residual_in.to(torch.float32)
        inv_rms = torch.rsqrt(pre.pow(2).mean(dim=-1, keepdim=True) + float(rms_eps))
        normed = (pre * inv_rms) * rms_gamma.to(torch.float32)
        pre_dtype = pre.to(input.dtype)
        normed_dtype = normed.to(input.dtype)
        if residual_out is not None:
            residual_out.copy_(pre_dtype)
        if norm_out is not None:
            norm_out.copy_(normed_dtype)
        return normed_dtype
    raise NotImplementedError(
        f"allreduce_fusion reference does not model pattern={pattern} "
        "(quantized / MoE patterns are multi-rank-only)"
    )


allreduce_fusion_trace = TraceTemplate(
    op_type="comm",
    name_prefix="allreduce_fusion",
    description=(
        "TRT-LLM / MNNVL fused AllReduce + (Residual + RMSNorm + optional "
        "FP8/FP4 Quantize). The reference models the fusion side of the op "
        "under the assumption that the input has already been reduced "
        "(AllReduce is a no-op single-rank); multi-rank correctness is "
        "exercised by tests/comm/."
    ),
    axes={
        "num_tokens": Var(description="Token count along dim 0."),
        "hidden_dim": Const(abbrev="h"),
    },
    inputs={
        "input": Tensor(
            ["num_tokens", "hidden_dim"],
            description="Pre-reduction token activations (this rank's shard).",
        ),
        "workspace": Scalar(
            "int64",
            description=(
                "AllReduceFusionWorkspace handle (opaque to the trace; "
                "its shape/content are backend-specific)."
            ),
        ),
        "pattern": Scalar(
            "int32",
            description=(
                "AllReduceFusionPattern enum: 0=AllReduce, "
                "1=AR+Residual+RMSNorm, 2..5=with FP8/FP4 quant, "
                "6..7=MoE reduction/finalize (trtllm-only)."
            ),
        ),
        "residual_in": Tensor(
            ["num_tokens", "hidden_dim"],
            optional=True,
            description="Residual to add (patterns 1..5).",
        ),
        "rms_gamma": Tensor(
            ["hidden_dim"],
            optional=True,
            description="RMSNorm weight (patterns 1..5).",
        ),
        "rms_eps": Scalar("float32", optional=True),
    },
    outputs={
        "output": Tensor(
            ["num_tokens", "hidden_dim"],
            dtype_from="input",
            description="Main output; semantics depend on pattern.",
        ),
    },
    tags=["status:verified", "stage:comm", "fused"],
    reference=_allreduce_fusion_reference,
)


# ── DCP all-to-all (context-parallel attention reduction) ────────────────────


@torch.no_grad()
def _decode_cp_a2a_alltoall_reference(
    partial_o: torch.Tensor,
    softmax_stats: torch.Tensor,
    workspace,
    cp_rank: int,
    cp_size: int,
    enable_pdl=None,
    **_unused,
):
    """Single-rank reference for the DCP all-to-all.

    The kernel is a multi-rank exchange: each rank sends its
    ``partial_o[..., peer, :]`` slice to the corresponding peer and
    receives the gathered contributions. In a single-process reference we
    return ``partial_o`` and ``softmax_stats`` unchanged — the trace
    captures the schema; multi-rank correctness is tested under
    ``tests/comm/``.
    """
    return partial_o.clone(), softmax_stats.clone()


decode_cp_a2a_alltoall_trace = TraceTemplate(
    op_type="comm",
    name_prefix="decode_cp_a2a_alltoall",
    description=(
        "Context-parallel attention all-to-all reduction. Each rank ships "
        "its ``partial_o[..., peer, :]`` slice to peer ``peer`` and "
        "receives all peers' contributions in return. Used during paged "
        "decode with context-parallelism. Single-rank reference is a "
        "passthrough; multi-rank correctness is exercised by tests/comm."
    ),
    axes={
        "batch_dim": Var(description="Leading batch dimension(s)."),
        "cp_size": Var(description="Context-parallel group size."),
        "head_dim": Const(abbrev="d"),
        "stats_dim": Const(
            description="Softmax stats trailing dim (>=2, even).", abbrev="s"
        ),
        "ws_elems_per_rank": Var(),
    },
    inputs={
        "partial_o": Tensor(
            ["batch_dim", "cp_size", "head_dim"],
            description="Per-rank partial attention outputs [..., cp_size, D].",
        ),
        "softmax_stats": Tensor(
            ["batch_dim", "cp_size", "stats_dim"],
            description="Per-rank softmax stats [..., cp_size, S].",
        ),
        "workspace": Tensor(["cp_size", "ws_elems_per_rank"], dtype="int64"),
        "cp_rank": Scalar("int32"),
        "cp_size": Scalar("int32"),
    },
    outputs={
        "partial_o_out": Tensor(
            ["batch_dim", "cp_size", "head_dim"], dtype_from="partial_o"
        ),
        "softmax_stats_out": Tensor(
            ["batch_dim", "cp_size", "stats_dim"], dtype_from="softmax_stats"
        ),
    },
    tags=["status:verified", "stage:comm"],
    reference=_decode_cp_a2a_alltoall_reference,
)
